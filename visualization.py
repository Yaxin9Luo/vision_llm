import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from models.models_v2l import VQModel_RoBERTa
from transformers import RobertaTokenizer
from omegaconf import OmegaConf
import yaml
import argparse
import numpy as np
from collections import Counter
import seaborn as sns

def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=1,
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
    parser.add_argument("--stage", type=int, default=1, help="Pretraining stage")
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
    parser.add_argument("--output_dir", default="./output_dir/frozen_roberta_codebook", help="path where to save, empty for no saving")
    parser.add_argument("--log_dir", default="./output_dir/frozen_roberta_codebook", help="path where to tensorboard log")
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
    parser.add_argument("--quantizer_type", type=str, default="org", help="Decoding Loss")

    parser.add_argument("--embed_dim", type=int, default=1024, help="Decoding Loss")
    parser.add_argument("--use_cblinear", type=int, default=0, help="Decoding Loss")
    parser.add_argument("--use_crossatt_dec", type=int, default=0, help="Decoding Loss")
    
    parser.add_argument("--disc_start", default=10000, type=int)
    parser.add_argument("--rate_q", type=float, default=1, help="Decoding Loss")
    parser.add_argument("--rate_p", type=float, default=0.1, help="VGG Loss")
    parser.add_argument("--rate_d", type=float, default=0.75, help="GAN Loss")


    return parser
def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config
def load_model(checkpoint_path, config_path, device, args):
    config = load_config(config_path)
    print("Loaded config:")
    print(config)

    print("Initializing VQModel_RoBERTa...")
    model = VQModel_RoBERTa(args=args, **config.model.params)
    print("Model structure:")
    print(model)
    
    print(f"Loading model from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    print("Keys in loaded state dict:")
    print(state_dict.keys())
    
    if 'model' in state_dict:
        state_dict = state_dict['model']
        print("Found 'model' key, using its content")
        print("Keys in model state dict:")
        print(state_dict.keys())
    
    print("Keys in current model state dict:")
    print(model.state_dict().keys())
    
    print("Attempting to load state dict...")
    try:
        model.load_state_dict(state_dict, strict=False)
        print("State dict loaded successfully with strict=False")
    except Exception as e:
        print(f"Error loading state dict: {e}")
    
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, device, target_size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)

def visualize_quantized_tokens(model, image_tensor, tokenizer,top_n_to_remove=5):
    with torch.no_grad():
        encoder_feature = model.quant_conv(model.encoder(image_tensor))
        quant, tk_labels, _ = model.quantize(encoder_feature)
    tk_labels = tk_labels.squeeze().cpu().numpy()
    vocab = tokenizer.get_vocab()
    id2word = {v: k for k, v in vocab.items()}
    
    # 检查 tk_labels 的维度
    if tk_labels.ndim == 1:
        # 如果是一维数组，将其重塑为二维数组
        side_length = int(tk_labels.shape[0] ** 0.5)
        tk_labels = tk_labels.reshape(side_length, side_length)
    
    words = [[id2word[int(label)] for label in row] for row in tk_labels]
    # 创建一个包含两个子图的图像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40, 20))
    
    # 左侧：tokens可视化
    im = ax1.imshow(tk_labels, cmap='tab20')
    # 每隔step个patch标注一次
    step=3
    for i in range(0, len(words), step):
        for j in range(0, len(words[i]), step):
            color_val = tk_labels[i, j]
            text_color = 'white' if color_val > np.mean(tk_labels) else 'black'
            ax1.text(j, i, words[i][j], 
                    ha='center', va='center',
                    color=text_color,
                    fontsize=14,
                    weight='bold',
                    rotation=45)
    
    ax1.set_title("Quantized Tokens Visualization (8x8 patches)")
    ax1.grid(True)
    
    # 右侧：原图
    # 将tensor转换回图像格式
    img = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # 反归一化
    img = img * 0.5 + 0.5
    img = np.clip(img, 0, 1)
    ax2.imshow(img)
    ax2.set_title("Original Image (128x128)")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('tokens_and_image_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

    return words, tokenizer.convert_tokens_to_string([word for row in words for word in row])
    # # 创建一个单词到数字的映射
    # unique_words = sorted(set(word for row in words for word in row))
    # word2num = {word: i for i, word in enumerate(unique_words)}
    # # 计算词频并移除最高频的词
    # # word_counts = Counter(word for row in words for word in row)
    # # most_common = [word for word, _ in word_counts.most_common(top_n_to_remove)]
    
    # # filtered_words = [[word for word in row if word not in most_common] for row in words]
    # # flattened_words = [word for row in filtered_words for word in row if word]
    # flattened_words = [word for row in words for word in row]
    # # 将单词列表连接成一个句子
    # sentence = tokenizer.convert_tokens_to_string(flattened_words)
    # # 将单词转换为数字
    # num_array = np.array([[word2num[word] for word in row] for row in words])
    
    # plt.figure(figsize=(20, 20))
    # im = plt.imshow(num_array, cmap='tab20')
    # plt.colorbar(im, label='Word Index')
    # plt.title("Quantized Tokens Visualization")
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig('quantized_tokens_visualization.png')
    # plt.close()

    # return words, sentence

def visualize_attention_scores(model, image_tensor, tokenizer, save_path='attention_visualization.png'):
    """
    Visualize attention scores from each layer of the LLM (RoBERTa) model
    """
    model.eval()
    with torch.no_grad():
        # Get image tokens
        encoder_feature = model.quant_conv(model.encoder(image_tensor))
        quant, tk_labels, _ = model.quantize(encoder_feature)
        
        # Reshape tk_labels to [batch_size, sequence_length]
        batch_size = 1
        tk_labels = tk_labels.view(batch_size, -1)
        
        # Forward pass with output_attentions=True
        outputs = model.llm(
            input_ids=tk_labels,
            output_attentions=True,
            return_dict=True
        )
        
        # Get attention scores from all layers
        attention_scores = outputs.attentions  # Tuple of tensors, one per layer
        
        # Create visualization
        n_layers = len(attention_scores)
        fig, axes = plt.subplots(4, 3, figsize=(20, 25))  # Adjust grid size based on number of layers
        axes = axes.flatten()
        
        for idx, attn in enumerate(attention_scores):
            if idx >= len(axes):
                break
                
            # Take the first head of the first batch for visualization
            attention_map = attn[0, 0].cpu().numpy()
            
            # Create heatmap
            sns.heatmap(attention_map, ax=axes[idx], cmap='viridis')
            axes[idx].set_title(f'Layer {idx+1} Attention')
            axes[idx].set_xlabel('Key tokens')
            axes[idx].set_ylabel('Query tokens')
        
        # Remove empty subplots if any
        for idx in range(len(attention_scores), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
def main():
    parser = get_args_parser()
    args = parser.parse_args()
    checkpoint_path = '/root/autodl-tmp/vision_llm/output_dir/frozen_roberta_codebook/vqgan_checkpoint-last.pth'
    config_path = '/root/autodl-tmp/vision_llm/vqgan_configs/v2l.yaml'
    image_path = '/root/autodl-tmp/ImageNet/train/n01614925/n01614925_47.JPEG'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_model(checkpoint_path, config_path, device, args)
    image_tensor = preprocess_image(image_path, device)
    
    tokenizer = RobertaTokenizer.from_pretrained('/root/autodl-tmp/roberta-large')
    visualize_attention_scores(model, image_tensor, tokenizer)
    print("Attention visualization saved as 'attention_visualization.png'")
    # words,sentence = visualize_quantized_tokens(model, image_tensor, tokenizer)
    
    # print("Visualization saved as 'quantized_tokens_visualization.png'")
    # print("Quantized tokens as a sentence:")
    # print(sentence)

if __name__ == "__main__":
    main()