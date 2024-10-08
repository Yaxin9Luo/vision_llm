import sys
import os
import requests

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from models.models_v2l import VQModel_LLaMA 
import yaml
import torch
import clip
from omegaconf import OmegaConf
import argparse
from transformers import GPT2Tokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config,GPT2Model
from PIL import Image
def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument("--max_seq_len", type=int, default=512, metavar="LENGTH", help="the maximum sequence length")

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)")

    parser.add_argument("--lr", type=float, default=4.5e-4, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr", type=float, default=0.0, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0"
    )

    parser.add_argument("--warmup_epochs", type=int, default=5, metavar="N", help="epochs to warmup LR")

    # Dataset parameters
    parser.add_argument("--output_dir", default="./output_dir", help="path where to save, empty for no saving")
    parser.add_argument("--log_dir", default="./output_dir", help="path where to tensorboard log")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', help='whether to use distributed training')
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    parser.add_argument("--imagenet_path", default="/root/autodl-tmp/ImageNet", type=str, help="imagenet_path")
    parser.add_argument("--vqgan_path", default="vqgan_weight/vqgan_imagenet_f16_16384", type=str, help="path of llama model")
    
    parser.add_argument("--n_vision_words", default=8192, type=int)
    parser.add_argument("--output_type", default="next_token_prediction", type=str, help="next_token_prediction/classification")
    parser.add_argument("--decode_rate", type=float, default=0, help="Decoding Loss")
    parser.add_argument("--n_class", default=1000, type=int)    
    parser.add_argument("--vq_config_path", type=str, default="vqgan_configs/v2l.yaml", help="Decoding Loss")
    parser.add_argument("--image_size", type=int, default=256, help="Decoding Loss")
    parser.add_argument("--stage", type=int, default=1, help="Decoding Loss")
    parser.add_argument("--quantizer_type", type=str, default="org", help="Decoding Loss")

    parser.add_argument("--embed_dim", type=int, default=1024, help="Decoding Loss")
    parser.add_argument("--use_cblinear", type=int, default=0, help="Decoding Loss")
    parser.add_argument("--use_crossatt_dec", type=int, default=0, help="Decoding Loss")
    
    parser.add_argument("--disc_start", default=10000, type=int)
    parser.add_argument("--rate_q", type=float, default=1, help="Decoding Loss")
    parser.add_argument("--rate_p", type=float, default=0.1, help="VGG Loss")
    parser.add_argument("--rate_d", type=float, default=0.75, help="GAN Loss")


    return parser
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config
def visualize_embeddings_similarity(model, image, top_k=10):
    # 获取VQGAN的codebook embeddings
    with torch.no_grad():
        encoder_feature = model.quant_conv(model.encoder(image))
        quant, _, [_, _, tk_labels] = model.encode(encoder_feature)
        
        # Apply random masking
        quant_masked, mask, ids_restore = model.random_masking(quant, mask_ratio=0.75)
        
        # Reshape quant for GPT-2 input
        N, C, H, W = quant_masked.shape
        quant_seq = quant_masked.permute(0, 2, 3, 1).reshape(N, H*W, C)
        
        # Pass through GPT-2
        gpt2_output = model.gpt2(inputs_embeds=quant_seq).last_hidden_state
        quant_pred = gpt2_output.reshape(N, H, W, C).permute(0, 3, 1, 2)
        quant = quant * (1 - mask.view(N, 1, H, W)) + quant_pred * mask.view(N, 1, H, W)
        dec = model.decode(quant)
        print(dec.shape)
        dec_img = dec.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        dec_img = (dec_img * imagenet_std + imagenet_mean) * 255
        dec_img = np.clip(dec_img, 0, 255).astype(np.uint8)
        dec_pil = Image.fromarray(dec_img)
        dec_pil.save('decoded_image.png')
        print("Decoded image saved as 'decoded_image.png'")
        exit()
    # # # 获取GPT-2 tokenizer
    # # gpt2_medium_path = "/root/.cache/huggingface/hub/models--gpt2-medium/snapshots/6dcaa7a952f72f9298047fd5137cd6e4f05f41da"
    # # tokenizer = GPT2Tokenizer.from_pretrained(gpt2_medium_path)
    # GPT2Model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    # GPT2Model.to(gpt2_output.device)
    # print(gpt2_output.shape)
    # logits = GPT2Model.lm_head(gpt2_output)
    # print(logits.shape)
    # predicted_token_ids = torch.argmax(logits, dim=-1)
    # print(predicted_token_ids[0,:128])
    # # Save predicted token IDs to a file
    # output_file = 'predicted_token_ids.txt'
    # with open(output_file, 'w') as f:
    #     for batch in predicted_token_ids:
    #         f.write(' '.join(map(str, batch.cpu().numpy())) + '\n')
    
    # print(f"Predicted token IDs saved to {output_file}")
    # exit()
    plt.figure(figsize=(15, 10))
    sns.heatmap(logits[0, :, :100].detach().cpu().numpy(), cmap='YlOrRd')
    plt.title('GPT-2 Output Logits (first 100 tokens)')
    plt.xlabel('Token ID')
    plt.ylabel('Sequence Position')
    plt.savefig('gpt2_output_logits.png')
    plt.close()
def main(args):
    img_path = '/root/autodl-tmp/vision_llm/mlruns/0/1a29b672e42a4ee9b38a088f0a29bba0/artifacts/recons_79_2.png'
    img = Image.open(img_path)
    img = img.resize((256, 256))
    img = np.array(img) / 255.


    assert img.shape == (256, 256, 3)

    # normalize by ImageNet mean and std
    img = img - imagenet_mean
    img = img / imagenet_std
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    plt.rcParams['figure.figsize'] = [5, 5]
    vqgan_chkpt_dir = '/root/autodl-tmp/mae/results/mae_vqgan_llm-decoder/checkpoint-20.pth'
    config = load_config("/root/autodl-tmp/vision_llm/vqgan_configs/v2l.yaml", display=True)
    model_vqgan = VQModel_LLaMA(args=args, **config.model.params)
    checkpoint = torch.load(vqgan_chkpt_dir, map_location='cpu')
    # Check the keys in the checkpoint
    print("Keys in checkpoint:")
    for key in checkpoint.keys():
        print(f"  {key}")
    
    # Get the model state dict
    state_dict = checkpoint['model']
    
    # Create a new state dict with matching keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # remove 'module.' prefix
        if k.startswith('encoder.'):
            new_state_dict[k] = v
        elif k.startswith('decoder.'):
            new_state_dict[k] = v
        elif k.startswith('quant_conv.'):
            new_state_dict[k] = v
        elif k.startswith('post_quant_conv.'):
            new_state_dict[k] = v
        elif k.startswith('tok_embeddings.'):
            new_state_dict[k] = v
        elif k.startswith('gpt2.'):
            new_state_dict[k] = v
        else:
            print(f"Unmatched key: {k}")
    
    # Load the state dict
    msg = model_vqgan.load_state_dict(new_state_dict, strict=False)
    # print("Load state dict message:", msg)
    
    model_vqgan = model_vqgan.to(device)
    model_vqgan.eval()
    
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    visualize_embeddings_similarity(model_vqgan, img_tensor)
    

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

