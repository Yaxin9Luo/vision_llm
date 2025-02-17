import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import importlib
from einops import rearrange
from torch.nn import Embedding
from models.encoder_decoder import Encoder, Decoder
from transformers import GPT2Model, GPT2Config,GPT2LMHeadModel
from transformers import RobertaModel, RobertaConfig
import torch.nn as nn

    
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class VQModel_LLaMA(pl.LightningModule):
    def __init__(self,
                 args,
                 ddconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.args = args
        embed_dim = args.embed_dim    
        self.quantize_type = args.quantizer_type
        self.e_dim = embed_dim
        self.remap = remap
        self.sane_index_shape = sane_index_shape
        ###Encoder & Decoder
        self.stage = args.stage
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        # self.llm_config = GPT2Config.from_pretrained('gpt2-medium')
        # self.llm = GPT2Model.from_pretrained('gpt2-medium', config=self.llm_config)
        # for param in self.llm.parameters():
        #     param.requires_grad = False
        #  # Use llm's embedding as the codebook and move it to the correct device
        # self.register_buffer('codebook', self.llm.wte.weight.detach())
        self.llm_config = GPT2Config.from_pretrained('gpt2-medium')
        self.llm = GPT2LMHeadModel.from_pretrained('gpt2-medium', config=self.llm_config)
        for param in self.llm.parameters():
            param.requires_grad = False
        # 使用llm的embedding作为codebook
        self.register_buffer('codebook', self.llm.transformer.wte.weight.detach())
 
    def quantize(self, z):

        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # Find nearest neighbors in the llm codebook
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
           torch.sum(self.codebook.detach()**2, dim=1) - 2 * \
           torch.einsum('bd,dn->bn', z_flattened, rearrange(self.codebook.detach(), 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, self.codebook).view(z.shape)
        codebook_loss = torch.mean((z_q.detach()-z)**2) + 0.33 * \
                torch.mean((z_q - z.detach()) ** 2)
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, min_encoding_indices, codebook_loss
    def forward(self, input, data_iter_step, step=0, is_val=False, k=21):
        
        encoder_feature = self.quant_conv(self.encoder(input))
        quant, tk_labels, codebook_loss = self.quantize(encoder_feature)
        
        N,C,H,W = quant.shape
        input_seq = quant.permute(0, 2, 3, 1).reshape(N, H*W, -1)  # [N, L, C]
        input_ids = tk_labels.reshape(N, -1)
        outputs = self.llm(
            input_ids=input_ids,
            labels=input_ids,
            output_hidden_states = True,
            return_dict=True
        )
        next_token_loss = outputs.loss
        hidden_states = outputs.hidden_states[-1]  # [B, L, C]
        # 重塑回图像格式并解码
        quant_pred = hidden_states.reshape(N, H, W, -1).permute(0, 3, 1, 2)  # [8, 1024, 16, 16]
        dec = self.decode(quant_pred)
        rec_loss = torch.mean(torch.abs(input.contiguous() - dec.contiguous()))
        loss = rec_loss + self.args.rate_q * codebook_loss
        if next_token_loss is not None:
            loss = loss + 0.1 * next_token_loss
        return loss, rec_loss, codebook_loss, next_token_loss, tk_labels, dec
        
    def encode(self, h):
        quant, indices = self.quantize(h)
        return quant, indices
    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
    def get_last_layer(self):
        return self.decoder.conv_out.weight



class VQModel_RoBERTa(pl.LightningModule):
    def __init__(self,
                 args,
                 ddconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,
                 ):
        super().__init__()
        self.image_key = image_key
        self.args = args
        embed_dim = args.embed_dim    
        self.quantize_type = args.quantizer_type
        self.e_dim = embed_dim
        self.remap = remap
        self.sane_index_shape = sane_index_shape
        ### Encoder & Decoder
        self.stage = args.stage
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        
        # RoBERTa Configuration
        self.llm_config = RobertaConfig.from_pretrained('FacebookAI/roberta-large')
        self.llm = RobertaModel.from_pretrained('FacebookAI/roberta-large', config=self.llm_config)
        for param in self.llm.parameters():
            param.requires_grad = False
        
        # 添加router相关配置
        self.use_mod = getattr(args, 'use_mod', True)
        self.capacity_factor = getattr(args, 'capacity_factor', 0.5)
        self.router_aux_loss_coef = getattr(args, 'router_aux_loss_coef', 0.01)
        
        # 添加router
        if self.use_mod:
            self.router = nn.Linear(embed_dim, 2, bias=False)
        
        # Use llm's embedding as the codebook
        self.register_buffer('codebook', self.llm.embeddings.word_embeddings.weight.detach())
        self.last_aux_loss = None  # 添加aux_loss保存

    def quantize(self, z):

        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # Find nearest neighbors in the llm codebook
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
           torch.sum(self.codebook.detach()**2, dim=1) - 2 * \
           torch.einsum('bd,dn->bn', z_flattened, rearrange(self.codebook.detach(), 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, self.codebook).view(z.shape)
        codebook_loss = torch.mean((z_q.detach()-z)**2) + 0.33 * \
                torch.mean((z_q - z.detach()) ** 2)
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, min_encoding_indices, codebook_loss

    def mod_masking(self, x):
        """使用router替代random masking"""
        N, C, H, W = x.shape
        L = H * W
        
        # 将输入reshape为序列形式以便router处理
        x_seq = x.permute(0, 2, 3, 1).reshape(N, L, C)
        
        # 使用router预测token的重要性
        router_logits = self.router(x_seq)  # [N, L, 2]
        route_probabilities = F.softmax(router_logits, dim=-1)[:, :, 1]
        
        # 选择top-k个token
        len_keep = int(L * self.capacity_factor)
        token_weights, ids_keep = torch.topk(route_probabilities, len_keep, dim=1)
        
        # 创建mask
        mask = torch.zeros(N, L, device=x.device)
        mask.scatter_(1, ids_keep, 1)
        
        # 应用mask
        x_masked = x.flatten(2) * mask.unsqueeze(1)
        x_masked = x_masked.view(N, C, H, W)
        
        # 计算router的辅助损失
        router_targets = torch.zeros_like(route_probabilities)
        router_targets.scatter_(1, ids_keep, 1)
        aux_loss = F.cross_entropy(router_logits.view(-1, 2), router_targets.view(-1).long())
        
        # 返回masked结果和辅助损失
        return x_masked, 1 - mask, ids_keep, aux_loss

    def forward(self, input, data_iter_step, step=0, is_val=False, k=21):
        encoder_feature = self.quant_conv(self.encoder(input))
        quant, tk_labels, codebook_loss = self.quantize(encoder_feature)
        
        # 使用router进行masking
        if self.use_mod and self.training:
            quant_masked, mask, ids_keep, aux_loss = self.mod_masking(quant)
            self.last_aux_loss = aux_loss  # 保存aux_loss供训练循环使用
        else:
            quant_masked, mask, ids_restore = self.random_masking(quant, mask_ratio=0.5)
            aux_loss = torch.tensor(0.0, device=quant.device)
            self.last_aux_loss = aux_loss
        
        # Reshape quant for RoBERTa input
        N, C, H, W = quant_masked.shape
        quant_seq = quant_masked.permute(0, 2, 3, 1).reshape(N, H*W, C)
        llm_output = self.llm(inputs_embeds=quant_seq).last_hidden_state

        if self.args.stage == 2:
            return quant, tk_labels, llm_output
            
        # Reshape RoBERTa output back to image shape
        quant_pred = llm_output.reshape(N, H, W, C).permute(0, 3, 1, 2)
        # Combine masked and predicted tokens
        quant = quant * (1 - mask.view(N, 1, H, W)) + quant_pred * mask.view(N, 1, H, W)
        dec = self.decode(quant)
        
        # Loss calculation
        rec_loss = torch.mean(torch.abs(input.contiguous() - dec.contiguous()))
        loss = rec_loss + self.args.rate_q * codebook_loss
        
        if self.use_mod and self.training:
            loss = loss + self.router_aux_loss_coef * aux_loss

        return loss, rec_loss, codebook_loss, tk_labels, dec

    def encode(self, h):
        quant, indices = self.quantize(h)
        return quant, indices
    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        # 正常的训练步骤
        x = self.get_input(batch, self.image_key)
        xrec, loss = self(x)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, loss = self(x)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(list(self.encoder.parameters()) +
                             list(self.decoder.parameters()) +
                             list(self.quant_conv.parameters()) +
                             list(self.post_quant_conv.parameters()),
                             lr=self.args.learning_rate)
        return opt