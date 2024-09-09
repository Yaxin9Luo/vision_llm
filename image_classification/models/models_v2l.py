import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import importlib
from einops import rearrange
from torch.nn import Embedding
from models.discriminator import NLayerDiscriminator, weights_init
import models.global_encoder as clip
from models.lpips import LPIPS
from models.encoder_decoder import Encoder, Decoder

    
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
        embed_dim = 768
        self.quantize_type = "org"
        self.e_dim = embed_dim
        self.remap = remap
        self.sane_index_shape = sane_index_shape
        ###Encoder & Decoder
        self.stage = 2
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)


        ####Codebook
        print("****Using Quantizer: %s"%(self.quantize_type))
        self.criterion = torch.nn.CrossEntropyLoss()
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        n_vision_words = 32000
        print("****Using Random Token Embedding****")
        print("Word Number:%d" %(n_vision_words))
        print("Feature Dim:%d" %(embed_dim))
        self.tok_embeddings = Embedding(n_vision_words, embed_dim)
        self.tok_embeddings.weight.data.uniform_(-1.0 / n_vision_words, 1.0 / n_vision_words)
        self.tok_embeddings.weight.requires_grad = True

        ####GAN & Perceptual Loss 
        self.discriminator = NLayerDiscriminator(input_nc=3,
                                n_layers=2,
                                use_actnorm=False,
                                ndf=64
                                ).apply(weights_init)
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = 0.1    

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
    
    def quantize(self, z, temp=None, rescale_logits=False, return_logits=False):

        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.args.use_cblinear == 1:
            tok_embeddings_weight = self.codebook_projection(self.tok_embeddings.weight)
        else:
            tok_embeddings_weight = self.tok_embeddings.weight


        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(tok_embeddings_weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(tok_embeddings_weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        
        min_encodings = None
        z_q = F.embedding(min_encoding_indices, tok_embeddings_weight).view(z.shape)
        loss = torch.mean((z_q.detach()-z)**2) + 0.33 * \
                torch.mean((z_q - z.detach()) ** 2)
    
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (d, min_encodings, min_encoding_indices)
    
    def forward(self, input, data_iter_step, step=0, is_val=False, k=21):
                   
        encoder_feature = self.quant_conv(self.encoder(input))
        
        quant, qloss, [_, _, tk_labels] = self.encode(encoder_feature)

        if self.stage == 2: ## if only needs quantized feature(eg. finetuning LLM) 
            return quant,tk_labels.view(input.shape[0], -1)

        dec = self.decode(quant)

        ###Loss
        rec_loss = torch.mean(torch.abs(input.contiguous() - dec.contiguous()))
        
        p_loss = torch.mean(self.perceptual_loss(input.contiguous(), dec.contiguous()))
        
        if step == 0: #Upadte Generator
            logits_fake = self.discriminator(dec)
            g_loss = -torch.mean(logits_fake)

            if is_val:
                loss = rec_loss + self.args.rate_q * qloss + self.perceptual_weight * p_loss + 0 * g_loss
                return loss, rec_loss, qloss, p_loss, g_loss, tk_labels.view(input.shape[0], -1), dec
            
            d_weight = self.calculate_adaptive_weight(rec_loss + self.perceptual_weight * p_loss, g_loss, self.args.rate_d, last_layer=self.decoder.conv_out.weight)
            
            if data_iter_step > self.args.disc_start:
                loss = rec_loss + self.args.rate_q * qloss + self.perceptual_weight * p_loss + d_weight * g_loss
            else:
                loss = rec_loss + self.args.rate_q * qloss + self.perceptual_weight * p_loss + 0 * g_loss

            return loss, rec_loss, qloss, p_loss, g_loss, tk_labels, dec
        else: #Upadte Discriminator
            logits_real =  self.discriminator(input.contiguous().detach().clone())
            logits_fake = self.discriminator(dec.detach().clone())
            d_loss = self.hinge_d_loss(logits_real, logits_fake)
            loss = d_loss + 0 * (rec_loss + qloss + p_loss)

            return loss, rec_loss, qloss, p_loss, d_loss, tk_labels, dec


    def encode(self, h):
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)

        return dec
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def decode_code(self, code_b):
        quant_b = self.quantize.embedding(code_b)
        dec = self.decode(quant_b)
        return dec