import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import importlib
from einops import rearrange
from torch.nn import Embedding
from models.encoder_decoder import Encoder, Decoder
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import RobertaModel, RobertaConfig
import torch.nn as nn
import numpy as np
import torchvision

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
        
        # Set default parameters
        if not hasattr(self.args, 'use_ar_recon'):
            self.args.use_ar_recon = False
        if not hasattr(self.args, 'train_ar_recon'):
            self.args.train_ar_recon = False
        if not hasattr(self.args, 'eval_ar_gen'):
            self.args.eval_ar_gen = False
            
        ### Encoder & Decoder
        self.stage = args.stage
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        
        # Use GPT2Model
        self.llm_config = GPT2Config.from_pretrained('gpt2-medium')
        self.llm = GPT2Model.from_pretrained('gpt2-medium', config=self.llm_config)
        
        # Unfreeze the last few layers for fine-tuning
        for i in range(len(self.llm.h) - 3, len(self.llm.h)):
            for param in self.llm.h[i].parameters():
                param.requires_grad = True
                
        # Use llm's embedding as codebook
        self.register_buffer('codebook', self.llm.wte.weight.detach())
        
        # Add position encoding for autoregressive prediction
        self.register_buffer('position_ids', torch.arange(0, 1024).expand((1, -1)))
        
        # Add projection layer for predicting the next patch embedding
        self.next_patch_predictor = nn.Linear(self.llm_config.hidden_size, self.e_dim)
        
        # Add embedding for the start token
        self.register_buffer('bos_embedding', torch.zeros(1, 1, self.e_dim))
        
        # Add temperature parameter for sampling
        self.temperature = 1.0
        

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
        
    def prepare_embeddings_for_ar(self, quant):
        """
        Prepare quantized patch embeddings for autoregressive sequence
        """
        # Convert quant from [B, C, H, W] to sequence [B, H*W, C]
        N, C, H, W = quant.shape
        quant_seq = quant.permute(0, 2, 3, 1).reshape(N, H*W, C)
        
        # Create input sequence and target sequence
        # Input sequence: [BOS, e1, e2, ..., en-1]
        # Target sequence: [e1, e2, ..., en]
        bos_tokens = self.bos_embedding.expand(N, -1, -1)
        input_seq = torch.cat([bos_tokens, quant_seq[:, :-1]], dim=1)
        target_seq = quant_seq
        
        return input_seq, target_seq
    
    def forward(self, input, data_iter_step=0, step=0, is_val=False, k=21):
        # Encode input image
        encoder_feature = self.quant_conv(self.encoder(input))
        quant, tk_labels, codebook_loss = self.quantize(encoder_feature)
        
        # Prepare autoregressive sequence
        input_seq, target_seq = self.prepare_embeddings_for_ar(quant)
        
        # Get position IDs
        seq_len = input_seq.shape[1]
        position_ids = self.position_ids[:, :seq_len]
        
        # Autoregressive prediction through GPT2
        llm_output = self.llm(inputs_embeds=input_seq, position_ids=position_ids).last_hidden_state
        
        # Predict next patch embedding
        pred_embeddings = self.next_patch_predictor(llm_output)
        
        if self.args.stage == 2:
            return quant, tk_labels, pred_embeddings
            
        # Calculate autoregressive loss (MSE loss)
        ar_loss = F.mse_loss(pred_embeddings, target_seq)
        
        # Reconstruct image from predicted embeddings
        N, L, C = pred_embeddings.shape
        H = W = int(np.sqrt(L))
        
        # Build complete image representation
        # By default, use original quant as base, only use predicted embeddings to evaluate autoregressive loss
        recon_embeddings = quant
        
        # If autoregressive reconstruction is enabled (during validation or training)
        if (is_val and self.args.use_ar_recon) or (not is_val and self.args.train_ar_recon):
            # Use autoregressive method to generate complete image representation
            # During training, we need to keep gradients; during validation, no gradients needed
            with torch.set_grad_enabled(not is_val):
                # Start generation from BOS
                generated = self.bos_embedding.expand(N, 1, -1).to(input.device)
                
                # Autoregressive generation
                for i in range(H*W):
                    # Get position IDs
                    pos_ids = self.position_ids[:, :generated.shape[1]]
                    
                    # Predict next embedding through GPT2
                    output = self.llm(inputs_embeds=generated, position_ids=pos_ids).last_hidden_state
                    next_emb = self.next_patch_predictor(output[:, -1:])
                    
                    # Add to generated sequence
                    generated = torch.cat([generated, next_emb], dim=1)
                
                # Remove BOS embedding
                generated = generated[:, 1:]
                
                # Reshape to image shape
                recon_embeddings = generated.reshape(N, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        # Decode reconstructed image
        dec = self.decode(recon_embeddings)
        
        # Calculate reconstruction loss
        rec_loss = torch.mean(torch.abs(input.contiguous() - dec.contiguous()))
        
        # Total loss
        loss = rec_loss + self.args.rate_q * codebook_loss + ar_loss
        
        return loss, rec_loss, codebook_loss,ar_loss, tk_labels, dec
        
    def encode(self, h):
        encoder_feature = self.quant_conv(self.encoder(h))
        quant, indices, _ = self.quantize(encoder_feature)
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
        x = self.get_input(batch, self.image_key)
        loss, rec_loss, codebook_loss, _, _ = self(x)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/rec_loss", rec_loss, prog_bar=True)
        self.log("train/codebook_loss", codebook_loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        loss, rec_loss, codebook_loss, _, dec = self(x, is_val=True)
        
        # Record standard losses
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/rec_loss", rec_loss, prog_bar=True)
        self.log("val/codebook_loss", codebook_loss, prog_bar=True)
        
        # If needed, evaluate autoregressive generation quality
        if batch_idx == 0 and hasattr(self.args, 'eval_ar_gen') and self.args.eval_ar_gen:
            # Generate images using autoregressive method
            ar_gen_img, _ = self.generate(input_image=x, temperature=0.0)  # No temperature, deterministic generation
            
            # Calculate reconstruction loss with original images
            ar_rec_loss = torch.mean(torch.abs(x.contiguous() - ar_gen_img.contiguous()))
            self.log("val/ar_rec_loss", ar_rec_loss, prog_bar=True)
            
            # If possible, save generated images for visualization
            if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'add_image'):
                # Convert images to range [0, 1]
                x_vis = (x + 1) / 2
                dec_vis = (dec + 1) / 2
                ar_gen_vis = (ar_gen_img + 1) / 2
                
                # Concatenate original images, standard reconstructions and autoregressive generations
                grid = torch.cat([x_vis[:4], dec_vis[:4], ar_gen_vis[:4]], dim=0)
                grid_img = torchvision.utils.make_grid(grid, nrow=4)
                
                # Add to tensorboard
                self.logger.experiment.add_image(
                    f'val/comparison_ep{self.current_epoch}', 
                    grid_img, 
                    global_step=self.global_step
                )
        
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(list(self.encoder.parameters()) +
                             list(self.decoder.parameters()) +
                             list(self.quant_conv.parameters()) +
                             list(self.post_quant_conv.parameters()) +
                             list(self.next_patch_predictor.parameters()) +
                             [p for i in range(len(self.llm.h) - 3, len(self.llm.h)) 
                              for p in self.llm.h[i].parameters() if p.requires_grad],
                             lr=self.args.learning_rate)
        return opt
        
    @torch.no_grad()
    def generate(self, input_image=None, start_embeddings=None, num_steps=None, temperature=1.0):
        """
        Autoregressive image generation
        """
        # If input image is provided, encode it
        if input_image is not None:
            quant, _ = self.encode(input_image)
            N, C, H, W = quant.shape
            seq_len = H * W
        # Otherwise use provided start embeddings
        elif start_embeddings is not None:
            N, L, C = start_embeddings.shape
            seq_len = L
            H = W = int(np.sqrt(seq_len))
        else:
            raise ValueError("Must provide either input_image or start_embeddings")
            
        # If number of steps not specified, use sequence length
        if num_steps is None:
            num_steps = seq_len
            
        # Create starting sequence (only BOS embedding)
        generated = self.bos_embedding.expand(N, 1, -1).to(input_image.device if input_image is not None else start_embeddings.device)
        
        # If start embeddings provided, add them to generated sequence
        if start_embeddings is not None:
            generated = torch.cat([generated, start_embeddings], dim=1)
        
        # Autoregressive generation
        for _ in range(num_steps):
            # Get position IDs
            position_ids = self.position_ids[:, :generated.shape[1]]
            
            # Predict next embedding through GPT2
            llm_output = self.llm(inputs_embeds=generated, position_ids=position_ids).last_hidden_state
            next_embedding = self.next_patch_predictor(llm_output[:, -1:])
            
            # Add noise to increase diversity
            if temperature > 0:
                noise = torch.randn_like(next_embedding) * temperature
                next_embedding = next_embedding + noise
            
            # Add to generated sequence
            generated = torch.cat([generated, next_embedding], dim=1)
            
            # Stop if generated sequence is long enough
            if generated.shape[1] > seq_len + 1:  # +1 is for BOS
                break
                
        # Remove BOS embedding
        generated = generated[:, 1:]
        
        # Reshape to image shape
        generated_embeddings = generated.reshape(N, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        # Decode generated image
        generated_image = self.decode(generated_embeddings)
        
        return generated_image, generated_embeddings
