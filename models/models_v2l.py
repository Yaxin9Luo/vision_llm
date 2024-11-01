import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import importlib
from einops import rearrange
from torch.nn import Embedding
from models.discriminator import NLayerDiscriminator, weights_init
from models.lpips import LPIPS
from models.encoder_decoder import Encoder, Decoder
from transformers import GPT2Model, GPT2Config,GPT2LMHeadModel
from transformers import RobertaModel, RobertaConfig

    
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
        ####GPerceptual Loss 
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = args.rate_p    
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
        p_loss = torch.mean(self.perceptual_loss(input.contiguous(), dec.contiguous()))
        loss = rec_loss + self.args.rate_q * codebook_loss + \
            self.perceptual_weight * p_loss + \
            0.1 * next_token_loss if next_token_loss is not None else 0.0
        return loss, rec_loss, codebook_loss, p_loss, next_token_loss, tk_labels, dec
        
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
        self.llm_config = RobertaConfig.from_pretrained('/root/autodl-tmp/roberta-large')
        self.llm = RobertaModel.from_pretrained('/root/autodl-tmp/roberta-large', config=self.llm_config)
        for param in self.llm.parameters():
            param.requires_grad = False
        # Use llm's embedding as the codebook and move it to the correct device
        self.register_buffer('codebook', self.llm.embeddings.word_embeddings.weight.detach())
        
        #### GAN & Perceptual Loss 
        self.discriminator = NLayerDiscriminator(input_nc=3,
                                    n_layers=2,
                                    use_actnorm=False,
                                    ndf=64
                                    ).apply(weights_init)
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = args.rate_p    

    def hinge_d_loss(self, logits_real, logits_fake):
        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss

    def calculate_adaptive_weight(self, nll_loss, g_loss, discriminator_weight, last_layer=None):

        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * discriminator_weight
        return d_weight
    
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
        p_loss = torch.mean(self.perceptual_loss(input.contiguous(), dec.contiguous()))
        if step == 0: #Upadte Generator
            logits_fake = self.discriminator(dec)
            g_loss = -torch.mean(logits_fake)
            if is_val:
                loss = rec_loss + self.args.rate_q * codebook_loss +self.perceptual_weight * p_loss + 0 * g_loss
                return loss, rec_loss,codebook_loss, p_loss, g_loss, tk_labels.view(input.shape[0], -1), dec
            
            d_weight = self.calculate_adaptive_weight(rec_loss + self.perceptual_weight * p_loss, g_loss, self.args.rate_d, last_layer=self.decoder.conv_out.weight)
            
            if data_iter_step > self.args.disc_start:
                loss = rec_loss  + self.args.rate_q * codebook_loss + self.perceptual_weight * p_loss + d_weight * g_loss
            else:
                loss = rec_loss + self.args.rate_q * codebook_loss + self.perceptual_weight * p_loss + 0 * g_loss
            return loss, rec_loss, codebook_loss, p_loss, g_loss, tk_labels, dec
        else: #Upadte Discriminator
            logits_real =  self.discriminator(input.contiguous().detach().clone())
            logits_fake = self.discriminator(dec.detach().clone())
            d_loss = self.hinge_d_loss(logits_real, logits_fake)
            loss = d_loss + 0 * (rec_loss  + p_loss)

            return loss, rec_loss, codebook_loss, p_loss, d_loss, tk_labels, dec
    def encode(self, h):
        quant, indices = self.quantize(h)
        return quant, indices
    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
    def get_last_layer(self):
        return self.decoder.conv_out.weight