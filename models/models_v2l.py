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
        
        if self.use_mod and self.training:
            quant_masked, mask, ids_keep, aux_loss = self.mod_masking(quant)
            self.last_aux_loss = aux_loss  
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
    

class VQModel_RoBERTa_mae(pl.LightningModule):
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
        # Use llm's embedding as the codebook and move it to the correct device
        self.register_buffer('codebook', self.llm.embeddings.word_embeddings.weight.detach())
        

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
    def random_masking(self, x, mask_ratio):
        N, C, H, W = x.shape
        L = H * W
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]

        # Flatten spatial dimensions
        x_flattened = x.flatten(2)

        # Create a mask for kept indices
        mask = torch.zeros(N, L, device=x.device)
        mask.scatter_(1, ids_keep, 1)

        # Apply masking
        x_masked = x_flattened * mask.unsqueeze(1)

        # Reshape back to original shape
        x_masked = x_masked.view(N, C, H, W)

        # Create binary mask (1 is remove, 0 is keep)
        mask = 1 - mask

        return x_masked, mask, ids_restore
    def forward(self, input, data_iter_step, step=0, is_val=False, k=21):
                   
        encoder_feature = self.quant_conv(self.encoder(input))
        quant, tk_labels, codebook_loss = self.quantize(encoder_feature)
        # Apply random masking
        quant_masked, mask, ids_restore = self.random_masking(quant, mask_ratio=0.5)
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
    

class MultiScale_VQModel_RoBERTa(pl.LightningModule):
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
        
        # 多尺度配置
        self.patch_sizes = getattr(args, 'patch_sizes', [16, 32])  # 默认使用16x16和8x8两种尺度
        self.patch_weights = getattr(args, 'patch_weights', None)  # 不同尺度的权重
        if self.patch_weights is None:
            self.patch_weights = {size: 1.0/len(self.patch_sizes) for size in self.patch_sizes}
        
        # 为每个尺度创建encoder和量化层
        self.encoders = nn.ModuleDict()
        self.quant_convs = nn.ModuleDict()
        
        for patch_size in self.patch_sizes:
            encoder_config = ddconfig.copy()
            if patch_size == 16:
                encoder_config.update({
                    'ch_mult': (1,2,2,4),  # 4次下采样，每次/2，总共/16
                    'num_resolutions': 4,
                })
            elif patch_size == 32:
                encoder_config.update({
                    'ch_mult': (1,2,2,2,4),    # 3次下采样，每次/2，总共/32
                    'num_resolutions': 5,
                })
            
            self.encoders[f'encoder_{patch_size}'] = Encoder(**encoder_config)
            self.quant_convs[f'quant_conv_{patch_size}'] = torch.nn.Conv2d(
                ddconfig["z_channels"], embed_dim, 1)
        
        # 共享的decoder
        self.decoder = Decoder(**ddconfig)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        
        # # RoBERTa配置
        self.llm_config = RobertaConfig.from_pretrained('FacebookAI/roberta-large')
        self.llm = RobertaModel.from_pretrained('FacebookAI/roberta-large', config=self.llm_config)
        for param in self.llm.parameters():
            param.requires_grad = False
        ## GPT2配置
        # self.llm_config = GPT2Config.from_pretrained('gpt2-medium')
        # self.llm = GPT2LMHeadModel.from_pretrained('gpt2-medium', config=self.llm_config)
        # for param in self.llm.parameters():
        #     param.requires_grad = False
        
        # Router配置
        self.use_mod = getattr(args, 'use_mod', True)
        self.capacity_factor = getattr(args, 'capacity_factor', 0.5)
        self.router_aux_loss_coef = getattr(args, 'router_aux_loss_coef', 0.01)
        
        if self.use_mod:
            self.routers = nn.ModuleDict({
                f'router_{size}': nn.Linear(embed_dim, 2, bias=False)
                for size in self.patch_sizes
            })
        
        # 使用LLM的embedding作为codebook
        self.register_buffer('codebook', self.llm.embeddings.word_embeddings.weight.detach()) # Roberta
        # self.register_buffer('codebook', self.llm.transformer.wte.weight.detach()) # GPT2
        self.last_aux_loss = None

    def multi_scale_quantize(self, features_dict):
        """处理多尺度特征的量化"""
        all_z_q = {}
        all_indices = {}
        total_codebook_loss = 0
        
        for patch_size, z in features_dict.items():
            z = rearrange(z, 'b c h w -> b h w c').contiguous()
            z_flattened = z.view(-1, self.e_dim)
            # 计算与codebook的距离
            d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(self.codebook.detach()**2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', z_flattened, 
                           rearrange(self.codebook.detach(), 'n d -> d n'))
                
            min_encoding_indices = torch.argmin(d, dim=1)
            z_q = F.embedding(min_encoding_indices, self.codebook).view(z.shape)
            
            # 计算codebook loss
            codebook_loss = torch.mean((z_q.detach()-z)**2) + 0.33 * \
                    torch.mean((z_q - z.detach()) ** 2)
            
            # Straight-through estimator
            z_q = z + (z_q - z).detach()
            z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
            
            if self.sane_index_shape:
                min_encoding_indices = min_encoding_indices.reshape(
                    z_q.shape[0], z_q.shape[2], z_q.shape[3])
                
            all_z_q[patch_size] = z_q
            all_indices[patch_size] = min_encoding_indices
            total_codebook_loss += codebook_loss * self.patch_weights[patch_size]
            
        return all_z_q, all_indices, total_codebook_loss

    def mod_masking(self, x, patch_size):
        """使用router进行masking,为每个尺度单独处理"""
        N, C, H, W = x.shape
        L = H * W
        
        x_seq = x.permute(0, 2, 3, 1).reshape(N, L, C)
        router = self.routers[f'router_{patch_size}']
        
        router_logits = router(x_seq)
        route_probabilities = F.softmax(router_logits, dim=-1)[:, :, 1]
        
        len_keep = int(L * self.capacity_factor)
        token_weights, ids_keep = torch.topk(route_probabilities, len_keep, dim=1)
        
        mask = torch.zeros(N, L, device=x.device)
        mask.scatter_(1, ids_keep, 1)
        
        x_masked = x.flatten(2) * mask.unsqueeze(1)
        x_masked = x_masked.view(N, C, H, W)
        
        router_targets = torch.zeros_like(route_probabilities)
        router_targets.scatter_(1, ids_keep, 1)
        aux_loss = F.cross_entropy(router_logits.view(-1, 2), 
                                 router_targets.view(-1).long())
        
        return x_masked, 1 - mask, ids_keep, aux_loss

    def random_masking(self, x, mask_ratio=0.5):
        """随机masking,用于验证阶段"""
        N, C, H, W = x.shape
        L = H * W
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        
        mask = torch.zeros(N, L, device=x.device)
        mask.scatter_(1, ids_keep, 1)
        
        x_masked = x.flatten(2) * mask.unsqueeze(1)
        x_masked = x_masked.view(N, C, H, W)
        
        return x_masked, 1 - mask, ids_restore

    def forward(self, input, data_iter_step, step=0, is_val=False, k=21):
        # print(f"\n==== Step {data_iter_step} ====")
        # print(f"Input shape: {input.shape}")  # 输入图像的尺寸

        # 1. 多尺度特征提取和量化
        features_dict = {}
        for patch_size in self.patch_sizes:
            encoder_out = self.encoders[f'encoder_{patch_size}'](input)
            quant_feature = self.quant_convs[f'quant_conv_{patch_size}'](encoder_out)
            features_dict[patch_size] = quant_feature
            # print(f"\nPatch size {patch_size}:")
            # print(f"- Encoder output shape: {encoder_out.shape}")
            # print(f"- Quant feature shape: {quant_feature.shape}")
        all_quant, all_tk_labels, codebook_loss = self.multi_scale_quantize(features_dict)
        
        # 2. 特征masking和准备
        combined_quant = []
        combined_labels = []
        total_aux_loss = 0
        
        for patch_size in self.patch_sizes:
            quant = all_quant[patch_size]
            tk_labels = all_tk_labels[patch_size]
            # print(f"\nPatch size {patch_size}:")
            # print(f"- Original quant shape: {quant.shape}")
            # print(f"- Token labels shape: {tk_labels.shape}")
            
            if self.use_mod and self.training:
                quant_masked, mask, _, aux_loss = self.mod_masking(quant, patch_size)
                total_aux_loss += aux_loss * self.patch_weights[patch_size]
            else:
                quant_masked, mask, _ = self.random_masking(quant, mask_ratio=0.5)
            
            N, C, H, W = quant_masked.shape
            # print(f"- Masked quant shape: {quant_masked.shape}")
            
            quant_seq = quant_masked.permute(0, 2, 3, 1).reshape(N, H*W, C)
            # print(f"- Reshaped to sequence: {quant_seq.shape}")
            combined_quant.append(quant_seq)
            combined_labels.append(tk_labels.reshape(N, -1))
        # 3. 合并特征并输入LLM
        combined_quant = torch.cat(combined_quant, dim=1)
        combined_labels = torch.cat(combined_labels, dim=1)
        # print(f"\nCombined features:")
        # print(f"- Combined quant shape: {combined_quant.shape}")
        # print(f"- Combined labels shape: {combined_labels.shape}")
        
        llm_output = self.llm(inputs_embeds=combined_quant).last_hidden_state
        # print(f"- LLM output shape: {llm_output.shape}")
        
        if self.args.stage == 2:
            return all_quant, all_tk_labels, llm_output
       # 4. 分离不同尺度的特征并重建
        start_idx = 0
        final_features = {}
        N = input.shape[0]
        
        feature_sizes = {
            16: 16,  # patch_size 16 实际输出 16x16
            32: 8    # patch_size 32 实际输出 8x8
        }
        
        for patch_size in self.patch_sizes:
            # 使用实际的特征图尺寸
            H = W = feature_sizes[patch_size]
            length = H * W
            scale_output = llm_output[:, start_idx:start_idx+length]
            # print(f"\nPatch size {patch_size}:")
            # print(f"- Feature size: {H}x{W}")
            # print(f"- Number of tokens: {length}")
            
            # 重新排列成特征图形状
            feat = scale_output.reshape(N, H, W, -1).permute(0, 3, 1, 2)
            # print(f"- Feature shape: {feat.shape}")
            
            # 将所有特征图调整到相同的空间分辨率（使用最大的分辨率16x16）
            if H != 16:  # 如果不是16x16，就上采样到16x16
                feat = F.interpolate(feat, size=(16, 16), 
                                  mode='bilinear', align_corners=False)
                # print(f"- After interpolation: {feat.shape}")
            
            final_features[patch_size] = feat
            start_idx += length
        
        # 加权融合不同尺度的特征
        final_quant = sum(self.patch_weights[size] * feat 
                         for size, feat in final_features.items())
        # print(f"\nFinal fusion:")
        # print(f"- Final quant shape: {final_quant.shape}")
        
        # 5. 解码重建
        dec = self.decode(final_quant)
        # 6. 损失计算
        rec_loss = torch.mean(torch.abs(input.contiguous() - dec.contiguous()))
        loss = rec_loss + self.args.rate_q * codebook_loss
        
        if self.use_mod and self.training:
            loss = loss + self.router_aux_loss_coef * total_aux_loss
            self.last_aux_loss = total_aux_loss
        
        return loss, rec_loss, codebook_loss, combined_labels, dec

    def encode(self, h):
        """编码函数，返回所有尺度的量化结果"""
        features_dict = {}
        for patch_size in self.patch_sizes:
            encoder_out = self.encoders[f'encoder_{patch_size}'](h)
            quant_feature = self.quant_convs[f'quant_conv_{patch_size}'](encoder_out)
            features_dict[patch_size] = quant_feature
            
        all_quant, all_indices, _ = self.multi_scale_quantize(features_dict)
        return all_quant, all_indices

    def decode(self, quant):
        """解码函数"""
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
        x = self.get_input(batch, self.image_key)
        loss, rec_loss, codebook_loss, _, _ = self(x)
        self.log('train/loss', loss)
        self.log('train/rec_loss', rec_loss)
        self.log('train/codebook_loss', codebook_loss)
        if self.last_aux_loss is not None:
            self.log('train/aux_loss', self.last_aux_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        loss, rec_loss, codebook_loss, _, _ = self(x, is_val=True)
        self.log('val/loss', loss)
        self.log('val/rec_loss', rec_loss)
        self.log('val/codebook_loss', codebook_loss)
        return loss

    def configure_optimizers(self):
        params_to_optimize = (
            list(self.encoders.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quant_convs.parameters()) +
            list(self.post_quant_conv.parameters())
        )
        
        if self.use_mod:
            params_to_optimize.extend(list(self.routers.parameters()))
            
        opt = torch.optim.Adam(params_to_optimize, lr=self.args.learning_rate)
        return opt
    @property
    def encoder(self):
        """为了兼容性提供encoder属性,返回所有尺度encoder的Sequential模块"""
        class MultiScaleEncoder(nn.Module):
            def __init__(self, encoders):
                super().__init__()
                self.encoders = encoders
                
            def parameters(self, recurse=True):
                # 返回所有encoder的参数
                for encoder in self.encoders.values():
                    for param in encoder.parameters(recurse=recurse):
                        yield param
                        
        return MultiScaleEncoder(self.encoders)

    @property
    def quant_conv(self):
        """为了兼容性提供quant_conv属性,返回所有尺度quant_conv的Sequential模块"""
        class MultiScaleQuantConv(nn.Module):
            def __init__(self, quant_convs):
                super().__init__()
                self.quant_convs = quant_convs
                
            def parameters(self, recurse=True):
                # 返回所有quant_conv的参数
                for quant_conv in self.quant_convs.values():
                    for param in quant_conv.parameters(recurse=recurse):
                        yield param
                        
        return MultiScaleQuantConv(self.quant_convs)
    




class VQModel_GPT_mae(pl.LightningModule):
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
        ### Encoder & Decoder
        self.stage = args.stage
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        
        self.llm_config = GPT2Config.from_pretrained('gpt2-medium')
        self.llm = GPT2Model.from_pretrained('gpt2-medium', config=self.llm_config)
        for param in self.llm.parameters():
            param.requires_grad = False
        # Use llm's embedding as the codebook and move it to the correct device
        self.register_buffer('codebook', self.llm.wte.weight.detach())
        

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
    def random_masking(self, x, mask_ratio):
        N, C, H, W = x.shape
        L = H * W
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]

        # Flatten spatial dimensions
        x_flattened = x.flatten(2)

        # Create a mask for kept indices
        mask = torch.zeros(N, L, device=x.device)
        mask.scatter_(1, ids_keep, 1)

        # Apply masking
        x_masked = x_flattened * mask.unsqueeze(1)

        # Reshape back to original shape
        x_masked = x_masked.view(N, C, H, W)

        # Create binary mask (1 is remove, 0 is keep)
        mask = 1 - mask

        return x_masked, mask, ids_restore
    def forward(self, input, data_iter_step, step=0, is_val=False, k=21):
                   
        encoder_feature = self.quant_conv(self.encoder(input))
        quant, tk_labels, codebook_loss = self.quantize(encoder_feature)
        # Apply random masking
        quant_masked, mask, ids_restore = self.random_masking(quant, mask_ratio=0.5)
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
    