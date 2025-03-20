import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import importlib
from einops import rearrange
from torch.nn import Embedding
from models.encoder_decoder import Encoder, Decoder
from transformers import GPT2Model, GPT2Config,GPT2LMHeadModel,RobertaModel, RobertaConfig, LlamaForCausalLM, LlamaConfig,AutoModel,AutoConfig
import torch.nn as nn
import math
    
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
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class VQVAE_LLM_Codebook(pl.LightningModule):
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
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        
        # LLM Configuration
        self.llm_config = AutoConfig.from_pretrained('gpt2-medium')
        self.llm = AutoModel.from_pretrained('gpt2-medium', config=self.llm_config)
        
        # use meta-llama/Llama-3.2-1B
        # self.llm_config =  AutoConfig.from_pretrained('Qwen/Qwen2.5-0.5B')
        # self.llm =  AutoModel.from_pretrained('Qwen/Qwen2.5-0.5B', config=self.llm_config)
        # Use llm's embedding as the codebook (freeze it)
        # shape: [vocab_size, e_dim]
        self.register_buffer('codebook', self.llm.wte.weight.detach())
        # codebook_embed_dim = self.codebook.shape[1]
        # self.proj_to_256 = nn.Linear(codebook_embed_dim, self.e_dim, bias=False)
        # delete llm to save memory
        del self.llm
        
        self.codebook_size = self.codebook.shape[0]
        
        # ------------------------------------------------------------------
        # 1) MLP adaptor instead of single linear layer
        #    e_dim ->  hidden size -> codebook_size
        # ------------------------------------------------------------------
        hidden_dim = int(self.e_dim // 2)
        self.adaptor = nn.Sequential(
            nn.Linear(self.e_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.e_dim),
            nn.LayerNorm(self.e_dim),
            nn.SiLU(),
            nn.Linear(self.e_dim, self.codebook_size),
        )
        self.adaptor.apply(init_weights)
        # ------------------------------------------------------------------
        # 2) Gumbel temperature annealing parameters
        # ------------------------------------------------------------------
        self.tau_start = 1.0          # initial temperature
        self.tau_min = 0.1            # minimum temperature
        self.tau_decay_rate = 1 # exponential decay base
        self.gumbel_temp = self.tau_start  # current temperature

        self.code_usage_counts = torch.zeros(self.codebook_size, dtype=torch.long)

    def quantize(self, z, tau=None):
        """
        Uses a Gumbel-Softmax approach to pick discrete indices for each latent vector.
        """
        if tau is None:
            tau = self.gumbel_temp

        # [B, C, H, W] -> [B, H, W, C]
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        b, h, w, c = z.shape
        z_flat = z.view(b * h * w, c)

        # Adaptor predicts logits -> [B*H*W, vocab_size]
        logits = self.adaptor(z_flat)

        # Gumbel-Softmax to get discrete one-hot
        one_hot = F.gumbel_softmax(logits, tau=tau, hard=True)
        # codebook_256 = self.proj_to_256(self.codebook)   # shape: [V, 256]
        # Quantized vectors via codebook
        # z_q_flat = one_hot @ codebook_256  # [B*H*W, e_dim]
        z_q_flat = one_hot @ self.codebook  # [B*H*W, e_dim]
        z_q = z_q_flat.view(b, h, w, c)
        
        # VQ-style loss
        codebook_loss = torch.mean((z_q.detach() - z)**2) + 0.33 * torch.mean((z_q - z.detach())**2)

        # Straight-through trick
        z_q = z + (z_q - z).detach()
        
        # Reshape back
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        # Discrete indices = argmax of one_hot
        min_encoding_indices = torch.argmax(one_hot, dim=-1)  # [B*H*W]
        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.view(b, h, w)

        return z_q, min_encoding_indices, codebook_loss

    def forward(self, input, data_iter_step, step=0, is_val=False, k=21):


        if not is_val:
            # Exponential decay: tau = max(tau_min, tau_start * (decay_rate ** step))
            # Here we use data_iter_step, but you can use 'step' or an internal self.current_step.
            self.gumbel_temp = max(
                self.tau_min,
                self.tau_start * (self.tau_decay_rate ** data_iter_step)
            )
            # or a linear-ish schedule across steps
            # self.gumbel_temp = max(
            #     self.tau_min,
            #     self.gumbel_temp * (1.0 - self.tau_decay)
            # )
        encoder_feature = self.quant_conv(self.encoder(input))
        
        quant, tk_labels, codebook_loss = self.quantize(encoder_feature, tau=self.gumbel_temp)

        # ~~~ Code Utilization Tracking ~~~
        # 1) Per-batch utilization - 计算每个图像中使用的不同token索引的总数
        # 首先确保tk_labels的形状是正确的
        if not self.sane_index_shape:
            # 从打印结果看，tk_labels的形状是[4096]
            # 假设批次大小为16，每个图像有256个token
            b = input.shape[0]  # 获取实际的批次大小
            tokens_per_image = tk_labels.shape[0] // b  # 计算每个图像的token数量
            
            # 重塑为[b, tokens_per_image]
            tk_labels = tk_labels.view(b, tokens_per_image)
        
        # 计算每个图像中唯一token的数量，然后计算平均值
        unique_tokens_per_image = []
        for i in range(tk_labels.shape[0]):
            unique_tokens_per_image.append(tk_labels[i].unique().numel())
        
        # 计算平均值
        unique_tokens_in_batch = sum(unique_tokens_per_image) / len(unique_tokens_per_image)
        
        # 2) Accumulate usage counts across the entire training
        self.update_code_usage(tk_labels)
        
        # Stage 2 just returns embeddings
        if self.args.stage == 2:
            return quant, tk_labels, unique_tokens_in_batch

        dec = self.decode(quant)
        rec_loss = torch.mean(torch.abs(input.contiguous() - dec.contiguous()))
        loss = rec_loss + self.args.rate_q * codebook_loss

        return loss, rec_loss, codebook_loss, tk_labels, dec, unique_tokens_in_batch

    @torch.no_grad()
    def update_code_usage(self, indices):
        """
        Accumulate usage counts across the training run.
        indices: discrete code indices from the batch, shape [B*H*W] or [B, H, W].
        """
        # Flatten to 1D
        flat_indices = indices.view(-1)
        
        # Move indices to CPU
        flat_indices = flat_indices.detach().cpu()
        
        # Index-add approach to increment usage counts
        # We add 1 for each occurrence of each index in flat_indices
        ones = torch.ones_like(flat_indices, dtype=torch.long)
        self.code_usage_counts.index_add_(0, flat_indices, ones)

    def on_train_epoch_end(self):
        """
        At the end of each epoch, log how many codes have been used so far.
        """
        used_codes = (self.code_usage_counts > 0).sum()
        self.log("unique_tokens_total", used_codes.item(), prog_bar=True, on_epoch=True)

    def encode(self, h):
        quant, indices, _ = self.quantize(h)
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