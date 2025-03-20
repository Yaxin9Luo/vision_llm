import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import importlib
from einops import rearrange
from torch.nn import Embedding
from models.encoder_decoder import Encoder, Decoder
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig,AutoModel
from transformers import RobertaModel, RobertaConfig
import torch.nn as nn
import numpy as np
import torchvision
import os

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


class Vision_LLM(pl.LightningModule):
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
        self.llm_config = GPT2Config.from_pretrained('gpt2-medium')
        # 从头初始化LLM，而不是加载预训练权重
        self.llm = GPT2LMHeadModel(config=self.llm_config)
        self.llm.init_weights()
        # 只加载预训练模型的词嵌入作为codebook
        pretrained_model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        pretrained_embeddings = pretrained_model.transformer.wte.weight.detach()
        del pretrained_model  # 释放预训练模型内存
        del self.decoder
        del self.post_quant_conv
        # 注册词嵌入作为codebook
        self.register_buffer('codebook', pretrained_embeddings)
        codebook_embed_dim = self.codebook.shape[1]
        # self.proj_to_256 = nn.Linear(codebook_embed_dim, self.e_dim, bias=False)
        #
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
        # ------------------------------------------------------------------
        # 2) Gumbel temperature annealing parameters
        # ------------------------------------------------------------------
        self.tau_start = 1.0          # initial temperature
        self.tau_min = 0.1            # minimum temperature
        self.tau_decay_rate = 1 # exponential decay base
        self.gumbel_temp = self.tau_start  # current temperature

        self.code_usage_counts = torch.zeros(self.codebook_size, dtype=torch.long)
        
        # 加载预训练模型（第一阶段）
        if ckpt_path is not None and os.path.exists(ckpt_path):
            self._load_pretrained_model(ckpt_path, ignore_keys)
            print(f"success to load the pretrained model: {ckpt_path}")
        
        # 冻结除LLM以外的所有参数
        self._freeze_non_llm_params()

    def _load_pretrained_model(self, ckpt_path, ignore_keys=[]):
        """加载预训练模型权重"""
        if os.path.isdir(ckpt_path):
            # 如果提供的是目录，查找最新的检查点文件
            checkpoint_files = [f for f in os.listdir(ckpt_path) if f.startswith('vqvae_checkpoint-') and f.endswith('.pth')]
            if not checkpoint_files:
                print(f"warning: no checkpoint file found in {ckpt_path}")
                return
            
            # 如果有last检查点，优先使用
            if 'vqvae_checkpoint-last.pth' in checkpoint_files:
                checkpoint_path = os.path.join(ckpt_path, 'vqvae_checkpoint-last.pth')
            else:
                # 否则使用编号最大的检查点
                checkpoint_files = [f for f in checkpoint_files if 'last' not in f]
                if not checkpoint_files:
                    print(f"warning: no valid checkpoint file found in {ckpt_path}")
                    return
                    
                # 按编号排序，获取最新的检查点
                checkpoint_files.sort(key=lambda x: int(x.split('-')[1].split('.')[0]))
                checkpoint_path = os.path.join(ckpt_path, checkpoint_files[-1])
        else:
            checkpoint_path = ckpt_path
        
        print(f"load the checkpoint file: {checkpoint_path}")
        try:
            # 添加weights_only=False来解决PyTorch 2.6的兼容性问题
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            print("success to load the checkpoint file using weights_only=False")
        except Exception as e:
            print(f"error to load the checkpoint file: {e}")
            try:
                # 尝试直接添加安全全局对象
                import argparse
                import torch.serialization
                torch.serialization.add_safe_globals([argparse.Namespace])
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                print("success to load the checkpoint file using add_safe_globals")
            except Exception as e2:
                print(f"try other methods to load the checkpoint file failed: {e2}")
                print("please check the checkpoint file.")
                return
        
        # 获取模型状态字典
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
            
        # 过滤不需要的键
        for k in list(state_dict.keys()):
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"delete the key: {k}")
                    del state_dict[k]
            
            # 跳过LLM相关的参数
            if k.startswith('llm.'):
                print(f"keep the LLM parameters for the second stage training: {k}")
                del state_dict[k]
        
        # 加载状态字典
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        
        print(f"success to load the pretrained model. missing keys: {missing_keys}, unexpected keys: {unexpected_keys}")


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
        z_q_flat = one_hot @ self.codebook
        z_q = z_q_flat.view(b, h, w, c)
        
        # VQ-style loss
        codebook_loss = torch.mean((z_q.detach() - z)**2) + 0.33 * torch.mean((z_q - z.detach())**2)

        # Straight-through trick
        z_q = z + (z_q - z).detach()
        
        # Reshape back
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        # Discrete indices = argmax of one_hot
        min_encoding_indices = torch.argmax(one_hot, dim=-1)  # [B*H*W]
        min_encoding_indices = min_encoding_indices.view(b, h*w)
        
        return z_q, min_encoding_indices, codebook_loss

    def forward(self, input, data_iter_step, step=0, is_val=False, k=21):
        b = input.shape[0]

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

        # 第二阶段：自回归LLM训练
        # 准备自回归训练的输入和目标
        # 获取LLM的输出
        outputs = self.llm(input_ids=tk_labels,labels=tk_labels,output_hidden_states=True)
        loss = outputs.loss
        
        return loss, tk_labels

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

    def _freeze_non_llm_params(self):
        """冻结除LLM以外的所有参数"""
        # 冻结编码器
        for param in self.encoder.parameters():
            param.requires_grad = False

        # for param in self.decoder.parameters():
        #     param.requires_grad = False
        
        # 冻结量化相关层
        for param in self.quant_conv.parameters():
            param.requires_grad = False

        # for param in self.post_quant_conv.parameters():
        #     param.requires_grad = False

        # # 冻结适配器
        for param in self.adaptor.parameters():
            param.requires_grad = False
        
        # 冻结投影层
        # for param in self.proj_to_256.parameters():
        #     param.requires_grad = False