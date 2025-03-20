import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from models.models_v2l import VQVAE_LLM_Codebook
from transformers import GPT2Tokenizer
from omegaconf import OmegaConf
import yaml
import argparse
import numpy as np
from collections import Counter
import seaborn as sns
import torch.nn.functional as F
from datasets import load_dataset
import clip
import albumentations
from torch.serialization import add_safe_globals
import os
import json

# 添加argparse.Namespace到安全全局对象列表
add_safe_globals([argparse.Namespace])

# 定义输出目录
OUTPUT_DIR = "/data/drstrange/yaxin/Projects/vision_llm/output_dir/visualization"
os.makedirs(OUTPUT_DIR, exist_ok=True)

##############################################################################
#                          HELPER FUNCTIONS                                  #
##############################################################################
def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=512, metavar="LENGTH")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=4.5e-4, metavar="LR")
    parser.add_argument("--blr", type=float, default=1e-3, metavar="LR")
    parser.add_argument("--min_lr", type=float, default=0.0, metavar="LR")
    parser.add_argument("--warmup_epochs", type=int, default=5, metavar="N")
    parser.add_argument("--output_dir", default="./output_dir/frozen_roberta_codebook")
    parser.add_argument("--log_dir", default="./output_dir/frozen_roberta_codebook")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Distributed
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")

    # Data & model paths
    parser.add_argument("--imagenet_path", default="/data/drstrange/yaxin/data/mini-imagenet", type=str)
    parser.add_argument("--vqgan_path", default="vqgan_weight/vqgan_imagenet_f16_16384", type=str)
    parser.add_argument("--n_vision_words", default=8192, type=int)
    parser.add_argument("--output_type", default="next_token_prediction", type=str)
    parser.add_argument("--decode_rate", type=float, default=0)
    parser.add_argument("--n_class", default=1000, type=int)
    parser.add_argument("--vq_config_path", type=str, default="vqgan_configs/v2l.yaml")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--quantizer_type", type=str, default="org")
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--use_cblinear", type=int, default=0)
    parser.add_argument("--use_crossatt_dec", type=int, default=0)
    parser.add_argument("--disc_start", default=10000, type=int)
    parser.add_argument("--rate_q", type=float, default=1)
    parser.add_argument("--rate_p", type=float, default=0.1)
    parser.add_argument("--rate_d", type=float, default=0.75)

    return parser

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_model(checkpoint_path, config_path, device, args):
    config = load_config(config_path)

    model = VQVAE_LLM_Codebook(args=args, **config.model.params)
    
    print(f"Loading model from {checkpoint_path}")
    try:
        # 尝试使用weights_only=True加载（默认行为）
        state_dict = torch.load(checkpoint_path, map_location=device)
        print("Successfully loaded model with weights_only=True")
    except Exception as e:
        print(f"Error with weights_only=True: {e}")
        print("Trying with weights_only=False...")
        # 如果失败，尝试使用weights_only=False
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print("Successfully loaded model with weights_only=False")
    
    if 'model' in state_dict:
        state_dict = state_dict['model']
        print("Found 'model' key, using its content")

    try:
        model.load_state_dict(state_dict, strict=False)
        print("State dict loaded successfully with strict=False")
    except Exception as e:
        print(f"Error loading state dict: {e}")
    
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image, device, target_size=(128, 128)):
    """
    Preprocess an image (PIL Image or numpy array) for the model.
    """
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 确保图像是PIL Image格式
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert('RGB')
    
    return transform(image).unsqueeze(0).to(device)

def load_imagenet_class_names(path="/data/drstrange/yaxin/data/mini-imagenet/class_names_with_labels.json"):
    """
    加载ImageNet类别名称映射
    
    Args:
        path: 类别名称映射文件路径，如果不存在则创建一个简单的映射
        
    Returns:
        class_names: 字典，键为类别ID，值为类别名称
    """
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                class_names = json.load(f)
            print(f"已加载 {len(class_names)} 个类别名称，从 {path}")
            
            # 检查是否需要转换类别ID格式
            # 如果键是形如 "n01532829" 的格式，我们需要创建一个数字索引到类别名称的映射
            if all(k.startswith('n') for k in list(class_names.keys())[:5]):
                # 加载类别ID到索引的映射
                class_id_to_idx = {}
                try:
                    # 尝试加载 class_names.json 文件获取映射关系
                    idx_to_id_path = os.path.join(os.path.dirname(path), "class_names.json")
                    if os.path.exists(idx_to_id_path):
                        with open(idx_to_id_path, 'r') as f:
                            idx_to_id = json.load(f)
                            # 反转映射关系
                            class_id_to_idx = {v: int(k) for k, v in idx_to_id.items()}
                            print(f"已加载类别ID到索引的映射，共 {len(class_id_to_idx)} 项")
                except Exception as e:
                    print(f"加载类别ID到索引的映射时出错: {e}")
                
                # 创建索引到类别名称的映射
                idx_to_name = {}
                for class_id, class_name in class_names.items():
                    if class_id in class_id_to_idx:
                        idx = class_id_to_idx[class_id]
                        idx_to_name[str(idx)] = class_name
                    
                if idx_to_name:
                    print(f"已创建索引到类别名称的映射，共 {len(idx_to_name)} 项")
                    return idx_to_name
                else:
                    print("无法创建索引到类别名称的映射，将使用原始类别ID映射")
            
            return class_names
        except Exception as e:
            print(f"从 {path} 加载类别名称时出错: {e}")
    
    # 如果文件不存在或加载失败，创建一个简单的映射
    print(f"在 {path} 未找到类别名称文件，创建简单映射")
    
    # 尝试从datasets库加载类别名称
    try:
        dataset = load_dataset("/data/drstrange/yaxin/data/mini-imagenet", split="train")
        if hasattr(dataset, 'features') and 'label' in dataset.features:
            if hasattr(dataset.features['label'], 'names'):
                names = dataset.features['label'].names
                class_names = {str(i): name for i, name in enumerate(names)}
                
                # 保存到文件以便将来使用
                try:
                    with open(path, 'w') as f:
                        json.dump(class_names, f)
                    print(f"已保存 {len(class_names)} 个类别名称到 {path}")
                except Exception as e:
                    print(f"保存类别名称到 {path} 时出错: {e}")
                
                return class_names
    except Exception as e:
        print(f"从数据集获取类别名称时出错: {e}")
    
    # 如果无法从数据集获取，则创建一个简单的映射
    return {str(i): f"class_{i}" for i in range(1000)}

def load_mini_imagenet_samples(num_samples=3, split="train", data_path="/data/drstrange/yaxin/data/mini-imagenet", max_categories=100):
    """
    加载mini-imagenet数据集的样本
    
    Args:
        num_samples: 每个类别加载的样本数量
        split: 数据集分割，可以是"train", "validation", "test"
        data_path: 数据集路径
        max_categories: 最多加载的类别数量，设为None则加载所有类别
        
    Returns:
        data_samples: 列表，每个元素为(image, category_name)
    """
    # 加载类别名称映射
    class_names = load_imagenet_class_names()
    
    # 加载数据集
    dataset = load_dataset(data_path, split=split)
    
    # 获取不同类别的样本
    samples_by_category = {}
    data_samples = []
    
    for idx, sample in enumerate(dataset):
        image = sample['image']
        label = sample['label']
        label_str = str(label)
        
        # 使用类别名称而不是数字标签
        category = class_names.get(label_str, f"class_{label_str}")
        
        if category not in samples_by_category:
            samples_by_category[category] = []
            
        if len(samples_by_category[category]) < num_samples:
            samples_by_category[category].append((image, category))
            
        # 如果已经收集了足够多的类别，则停止
        if max_categories is not None and len(samples_by_category) >= max_categories:
            if all(len(samples) >= num_samples for samples in samples_by_category.values()):
                break
    
    # 从每个类别中选择样本
    for category, samples in samples_by_category.items():
        data_samples.extend(samples[:num_samples])
    
    print(f"Loaded {len(data_samples)} samples from {len(samples_by_category)} categories")
    return data_samples

##############################################################################
#                      VISUALIZATION / ANALYSIS FUNCTIONS                    #
##############################################################################
def visualize_quantized_tokens(model, image_tensor, tokenizer, save_path='tokens_and_image_visualization.png'):
    """
    1) Obtain discrete token indices from the model quantizer.
    2) Visualize them (alongside the original image).
    3) Return the 2D token grid (as strings) and also a flattened sentence.
    """
    # 确保保存路径在指定目录下
    save_path = os.path.join(OUTPUT_DIR, save_path)
    
    with torch.no_grad():
        encoder_feature = model.quant_conv(model.encoder(image_tensor))
        quant, tk_labels, _ = model.quantize(encoder_feature)
    
    # Move to CPU numpy
    tk_labels = tk_labels.squeeze().cpu().numpy()

    # Build id2word mapping
    vocab = tokenizer.get_vocab()
    id2word = {v: k for k, v in vocab.items()}
    
    # If your codebook indices differ from the tokenizer's internal IDs,
    # you might need a custom mapping. But here we assume 1-to-1 for demonstration.
    
    # Reshape to a square if needed
    if tk_labels.ndim == 1:
        side_length = int(tk_labels.shape[0] ** 0.5)
        tk_labels = tk_labels.reshape(side_length, side_length)
    
    words_2d = [[id2word[int(label)] for label in row] for row in tk_labels]

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: token index heatmap
    im = ax1.imshow(tk_labels, cmap='tab20')
    ax1.set_title("Quantized Token Indices")
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Right: original image
    img = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = img * 0.5 + 0.5  # de-normalize
    img = np.clip(img, 0, 1)
    ax2.imshow(img)
    ax2.set_title("Original Image")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    # Flatten tokens for a "sentence" (Roberta doesn't map all to readable words, but it's illustrative)
    flattened_tokens = [w for row in words_2d for w in row]
    sentence = tokenizer.convert_tokens_to_string(flattened_tokens)
    return words_2d, sentence

def visualize_token_distribution_by_category(category_token_usage, tokenizer, save_path='token_distribution_by_category.png'):
    """
    可视化不同类别的token分布情况
    
    Args:
        category_token_usage: 字典，键为类别名称，值为Counter对象（token索引 -> 频率）
        tokenizer: tokenizer对象，用于将token索引转换为token字符串
        save_path: 保存路径
    """
    # 确保保存路径在指定目录下
    save_path = os.path.join(OUTPUT_DIR, save_path)
    
    # 获取所有类别和所有token
    categories = list(category_token_usage.keys())
    
    # 为每个类别找出最常见的token
    top_tokens_by_category = {}
    all_top_tokens = set()
    top_k = 20  # 每个类别显示的top token数量
    
    # Build reverse vocab
    vocab = tokenizer.get_vocab()
    id2word = {v: k for k, v in vocab.items()}
    
    for cat, counter in category_token_usage.items():
        most_common = counter.most_common(top_k)
        top_tokens = [token_idx for token_idx, _ in most_common]
        top_tokens_by_category[cat] = top_tokens
        all_top_tokens.update(top_tokens)
    
    # 将所有top token转换为列表并排序
    all_top_tokens = sorted(list(all_top_tokens))
    
    # 创建一个矩阵，行为类别，列为token
    matrix = np.zeros((len(categories), len(all_top_tokens)))
    
    for i, cat in enumerate(categories):
        counter = category_token_usage[cat]
        for j, token_idx in enumerate(all_top_tokens):
            matrix[i, j] = counter[token_idx]
    
    # 归一化每一行
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, row_sums, where=row_sums!=0)
    
    # 创建token标签
    token_labels = [f"{id2word.get(idx, f'UNK_{idx}')} ({idx})" for idx in all_top_tokens]
    
    # 可视化
    plt.figure(figsize=(max(30, len(all_top_tokens) * 0.8), max(8, len(categories) * 0.5)))
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlGnBu',
                xticklabels=token_labels, yticklabels=categories)
    plt.title('Token Distribution by Category')
    plt.xlabel('Tokens')
    plt.ylabel('Categories')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved token distribution visualization to {save_path}")
    
    # 另外创建一个条形图，显示每个类别的top tokens
    n_cats = len(categories)
    fig, axes = plt.subplots(n_cats, 1, figsize=(10, 4 * n_cats))
    if n_cats == 1:
        axes = [axes]
    
    for i, cat in enumerate(categories):
        counter = category_token_usage[cat]
        most_common = counter.most_common(top_k)
        tokens = [id2word.get(idx, f'UNK_{idx}') for idx, _ in most_common]
        freqs = [freq for _, freq in most_common]
        
        axes[i].bar(tokens, freqs)
        axes[i].set_title(f'Top {top_k} Tokens for Category: {cat}')
        axes[i].set_ylabel('Frequency')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    bar_save_path = save_path.replace('.png', '_bar.png')
    plt.savefig(bar_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved token frequency bar charts to {bar_save_path}")

##############################################################################
#                           ANALYSIS / HYPOTHESIS TEST                       #
##############################################################################
def analyze_token_frequency_per_category(model, tokenizer, device, data_samples, top_k=5):
    """
    data_samples: list of (image, category_name) 
        e.g. [(image1, 'dog'), (image2, 'cat'), ...]
    top_k: how many top tokens to display for each category

    This function:
      - Loads each image
      - Gets token indices from model.quantize
      - Accumulates token usage in a Counter per category
      - Finally prints top_k tokens for each category
      
    Returns:
      - category_token_usage: 字典，键为类别名称，值为Counter对象（token索引 -> 频率）
    """
    # Maps category -> Counter of token usage
    category_token_usage = {}

    # Build reverse vocab
    vocab = tokenizer.get_vocab()
    id2word = {v: k for k, v in vocab.items()}

    for (img, cat) in data_samples:
        # Preprocess & forward
        img_tensor = preprocess_image(img, device)
        with torch.no_grad():
            enc_feat = model.quant_conv(model.encoder(img_tensor))
            _, tk_labels, _ = model.quantize(enc_feat)
        
        # Flatten
        tk_labels = tk_labels.view(-1).cpu().numpy()
        
        # Initialize category if not exist
        if cat not in category_token_usage:
            category_token_usage[cat] = Counter()
        
        # Update usage
        for idx in tk_labels:
            category_token_usage[cat][idx] += 1
    
    # Now print top_k tokens for each category
    for cat, counter in category_token_usage.items():
        print(f"\n=== Category: {cat} ===")
        most_common = counter.most_common(top_k)
        for token_idx, freq in most_common:
            token_str = id2word.get(token_idx, f"[UNK_{token_idx}]")
            print(f"   Token: {token_str:<15}  Frequency: {freq}")
    
    return category_token_usage

def load_llm_model(model_name, device):
    """
    加载单独的LLM模型用于分析
    
    Args:
        model_name: 模型名称或路径
        device: 设备
        
    Returns:
        model: 加载的LLM模型
    """
    from transformers import GPT2LMHeadModel
    
    print(f"Loading LLM model: {model_name}")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model

def analyze_token_usage(vqvae_model, tokenizer, device, data_samples):
    # Maps category -> Counter of token usage
    category_token_usage = {}

    # Build reverse vocab
    vocab = tokenizer.get_vocab()
    id2word = {v: k for k, v in vocab.items()}

    for (img, cat) in data_samples:
        # Preprocess & forward through VQVAE
        img_tensor = preprocess_image(img, device)
        with torch.no_grad():
            enc_feat = vqvae_model.quant_conv(vqvae_model.encoder(img_tensor))
            _, tk_labels, _ = vqvae_model.quantize(enc_feat)
        
        # Flatten
        tk_labels = tk_labels.view(1, -1)  # [1, seq_len]
        
        # 将token索引转换为numpy数组
        tk_labels_np = tk_labels.view(-1).cpu().numpy()
        
        # Initialize category if not exist
        if cat not in category_token_usage:
            category_token_usage[cat] = Counter()
        
        # Update usage
        for idx in tk_labels_np:
            category_token_usage[cat][idx] += 1
    
    return category_token_usage

def visualize_multiple_images_with_tokens(model, data_samples, tokenizer, device, num_images=10, save_path='multiple_images_visualization.png', group_by_category=True, enhance_visualization=True):
    """
    批量可视化多张图片及其量化后的句子，每张图片单独保存
    
    Args:
        model: VQVAE模型
        data_samples: 数据样本列表，每个元素为(image, category_name)
        tokenizer: tokenizer对象
        device: 设备
        num_images: 要可视化的图片数量
        save_path: 保存路径
        group_by_category: 是否按类别分组展示
        enhance_visualization: 是否增强可视化效果
    """
    # 确保保存路径在指定目录下
    save_path = os.path.join(OUTPUT_DIR, save_path)
    
    # 如果按类别分组，则重新组织数据样本
    if group_by_category:
        # 按类别分组
        samples_by_category = {}
        for img, category in data_samples:
            if category not in samples_by_category:
                samples_by_category[category] = []
            samples_by_category[category].append((img, category))
        
        # 从每个类别中选择样本
        samples_to_visualize = []
        for category, samples in samples_by_category.items():
            # 每个类别最多选择3个样本
            samples_to_visualize.extend(samples[:min(3, len(samples))])
            if len(samples_to_visualize) >= num_images:
                break
        
        # 如果样本数量不足，则从剩余样本中补充
        if len(samples_to_visualize) < num_images:
            remaining_samples = [s for s in data_samples if s not in samples_to_visualize]
            samples_to_visualize.extend(remaining_samples[:num_images - len(samples_to_visualize)])
    else:
        # 直接限制图片数量
        samples_to_visualize = data_samples[:num_images]
    
    num_samples = len(samples_to_visualize)
    
    # 创建一个大图，包含所有样本
    fig, axes = plt.subplots(num_samples, 4, figsize=(24, 6 * num_samples))
    
    # 如果只有一张图片，确保axes是二维的
    if num_samples == 1:
        axes = np.array([axes])
    
    # Build id2word mapping
    vocab = tokenizer.get_vocab()
    id2word = {v: k for k, v in vocab.items()}
    
    sentences = []
    
    # 创建一个文本文件，保存所有句子
    txt_save_path = save_path.replace('.png', '_sentences.txt')
    
    # 首先收集所有图片中出现的token，为它们创建一个全局颜色映射
    all_tokens = set()
    for img, _ in samples_to_visualize:
        img_tensor = preprocess_image(img, device)
        with torch.no_grad():
            encoder_feature = model.quant_conv(model.encoder(img_tensor))
            _, tk_labels, _ = model.quantize(encoder_feature)
        tk_labels_np = tk_labels.squeeze().cpu().numpy()
        all_tokens.update(np.unique(tk_labels_np))
    
    # 创建一个从token ID到颜色的映射
    # 使用一个有足够变化的颜色映射，如果token数量超过colormap的范围，我们将循环使用颜色
    cmap = plt.cm.get_cmap('tab20', 20)  # 基础colormap
    token_to_color = {}
    for i, token_id in enumerate(sorted(all_tokens)):
        # 循环使用颜色，但添加一些变化以区分
        color_idx = i % 20
        base_color = np.array(cmap(color_idx))
        # 对于超出20的token，稍微调整颜色的亮度
        if i >= 20:
            # 调整亮度，但保持在有效范围内
            brightness_factor = 0.7 + (0.3 * ((i // 20) % 3))  # 在0.7-1.0之间循环
            base_color[:3] = np.clip(base_color[:3] * brightness_factor, 0, 1)
        token_to_color[token_id] = base_color
    
    # 创建一个自定义的colormap函数，用于imshow
    def token_cmap(token_id):
        return token_to_color.get(token_id, [0, 0, 0, 1])  # 默认黑色
    
    with open(txt_save_path, 'w') as f:
        for i, (img, category) in enumerate(samples_to_visualize):
            # 预处理图片
            img_tensor = preprocess_image(img, device)
            
            # 获取量化token
            with torch.no_grad():
                encoder_feature = model.quant_conv(model.encoder(img_tensor))
                quant, tk_labels, _ = model.quantize(encoder_feature)
                
                # 使用decoder重建图像
                reconstructed_img = model.decode(quant)
            
            # 移到CPU并转为numpy
            tk_labels_np = tk_labels.squeeze().cpu().numpy()
            
            # 如果需要，重塑为方形
            if tk_labels_np.ndim == 1:
                side_length = int(tk_labels_np.shape[0] ** 0.5)
                tk_labels_np = tk_labels_np.reshape(side_length, side_length)
            
            # 创建2D单词网格
            words_2d = [[id2word[int(label)] for label in row] for row in tk_labels_np]
            
            # 展平tokens为"句子"
            flattened_tokens = [w for row in words_2d for w in row]
            sentence = tokenizer.convert_tokens_to_string(flattened_tokens)
            sentences.append(sentence)
            
            # 写入文本文件
            f.write(f"Image {i+1}, category: {category}\n")
            f.write(f"sentence: {sentence}\n\n")
            
            # 左图：原始图片
            img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img_np = img_np * 0.5 + 0.5  # 反归一化
            img_np = np.clip(img_np, 0, 1)
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title(f"original image: {category}", fontsize=12)
            axes[i, 0].axis('off')
            
            # 第二图：重建图像
            recon_img_np = reconstructed_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            recon_img_np = recon_img_np * 0.5 + 0.5  # 反归一化
            recon_img_np = np.clip(recon_img_np, 0, 1)
            axes[i, 1].imshow(recon_img_np)
            axes[i, 1].set_title("reconstructed image", fontsize=12)
            axes[i, 1].axis('off')
            
            # 计算原图和重建图像的MSE和PSNR
            mse = np.mean((img_np - recon_img_np) ** 2)
            psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 100
            
            # 在重建图像下方添加MSE和PSNR信息
            axes[i, 1].text(0.5, -0.1, f"MSE: {mse:.4f}, PSNR: {psnr:.2f} dB", 
                          transform=axes[i, 1].transAxes, ha='center', fontsize=10)
            
            # 第三图：token索引热图 - 使用全局颜色映射
            # 创建一个彩色图像数组，每个像素对应一个token的颜色
            height, width = tk_labels_np.shape
            colored_tokens = np.zeros((height, width, 4))
            for y in range(height):
                for x in range(width):
                    token_id = tk_labels_np[y, x]
                    colored_tokens[y, x] = token_to_color.get(token_id, [0, 0, 0, 1])
            
            # 显示彩色token图
            axes[i, 2].imshow(colored_tokens)
            axes[i, 2].set_title("quantized token indices", fontsize=12)
            axes[i, 2].set_xticks([])
            axes[i, 2].set_yticks([])
            
            # 为热图上的不同token添加文本标签
            # 找出唯一的token索引及其位置
            unique_tokens = np.unique(tk_labels_np)
            print(f"unique_tokens: {unique_tokens}")
            max_labels = len(unique_tokens)
            for idx, token_idx in enumerate(unique_tokens[:max_labels]):
                # 找到该token的一个位置
                positions = np.argwhere(tk_labels_np == token_idx)
                if len(positions) > 0:
                    # 选择该token的中心位置
                    center_pos = positions[len(positions)//2]
                    y, x = center_pos
                    token_text = id2word.get(int(token_idx), f"UNK_{token_idx}")
                    # 添加文本标签，白色背景增加可读性
                    axes[i, 2].text(x, y, token_text, color='black', fontsize=8,
                                  ha='center', va='center', 
                                  bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
            # 第四图：文本描述
            axes[i, 3].text(0.05, 0.5, sentence[:200] + "..." if len(sentence) > 200 else sentence, 
                          wrap=True, fontsize=10, va='center')
            axes[i, 3].set_title("generated sentence", fontsize=12)
            axes[i, 3].axis('off')
            
            # 为每张图片单独保存可视化结果
            if enhance_visualization:
                # 创建一个更大的图，包含更多信息
                single_fig = plt.figure(figsize=(24, 12))
                
                # 定义网格布局
                gs = single_fig.add_gridspec(2, 4)
                
                # 原始图像
                ax1 = single_fig.add_subplot(gs[0, 0])
                ax1.imshow(img_np)
                ax1.set_title(f"original image: {category}", fontsize=14)
                ax1.axis('off')
                
                # 重建图像
                ax2 = single_fig.add_subplot(gs[0, 1])
                ax2.imshow(recon_img_np)
                ax2.set_title(f"reconstructed image (MSE: {mse:.4f}, PSNR: {psnr:.2f} dB)", fontsize=14)
                ax2.axis('off')
                
                # 差异图
                ax3 = single_fig.add_subplot(gs[0, 2])
                diff = np.abs(img_np - recon_img_np)
                ax3.imshow(diff, cmap='hot')
                ax3.set_title("reconstruction error (absolute difference)", fontsize=14)
                ax3.axis('off')
                
                # Token索引热图 - 使用全局颜色映射
                ax4 = single_fig.add_subplot(gs[0, 3])
                # 创建彩色token图
                ax4.imshow(colored_tokens)
                ax4.set_title("quantized token indices", fontsize=14)
                ax4.set_xticks([])
                ax4.set_yticks([])
                
                # 为热图上的不同token添加文本标签
                unique_tokens = np.unique(tk_labels_np)
                max_labels = len(unique_tokens)
                for idx, token_idx in enumerate(unique_tokens[:max_labels]):
                    # 找到该token的一个位置
                    positions = np.argwhere(tk_labels_np == token_idx)
                    if len(positions) > 0:
                        # 选择该token的中心位置
                        center_pos = positions[len(positions)//2]
                        y, x = center_pos
                        token_text = id2word.get(int(token_idx), f"UNK_{token_idx}")
                        # 添加文本标签，白色背景增加可读性
                        ax4.text(x, y, token_text, color='black', fontsize=8,
                                ha='center', va='center', 
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                
                # 生成的句子
                ax5 = single_fig.add_subplot(gs[1, :2])
                ax5.text(0.05, 0.5, sentence, wrap=True, fontsize=12, va='center')
                ax5.set_title("generated sentence", fontsize=14)
                ax5.axis('off')
                
                # 查找同类别的其他图像进行对比
                similar_images = []
                for other_img, other_cat in data_samples:
                    if other_cat == category and (other_img, other_cat) != (img, category):
                        similar_images.append((other_img, other_cat))
                        if len(similar_images) >= 2:  # 最多找2个相似图像
                            break
                
                # 显示相似图像
                if similar_images:
                    ax6 = single_fig.add_subplot(gs[1, 2])
                    similar_img, _ = similar_images[0]
                    similar_img_tensor = preprocess_image(similar_img, device)
                    similar_img_np = similar_img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    similar_img_np = similar_img_np * 0.5 + 0.5
                    similar_img_np = np.clip(similar_img_np, 0, 1)
                    ax6.imshow(similar_img_np)
                    ax6.set_title(f"same category image 1", fontsize=14)
                    ax6.axis('off')
                    
                    if len(similar_images) > 1:
                        ax7 = single_fig.add_subplot(gs[1, 3])
                        similar_img, _ = similar_images[1]
                        similar_img_tensor = preprocess_image(similar_img, device)
                        similar_img_np = similar_img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        similar_img_np = similar_img_np * 0.5 + 0.5
                        similar_img_np = np.clip(similar_img_np, 0, 1)
                        ax7.imshow(similar_img_np)
                        ax7.set_title(f"same category image 2", fontsize=14)
                        ax7.axis('off')
            else:
                # 使用原来的简单布局
                single_fig, single_axes = plt.subplots(1, 4, figsize=(24, 6))
                
                # 左图：原始图片
                single_axes[0].imshow(img_np)
                single_axes[0].set_title(f"original image: {category}", fontsize=12)
                single_axes[0].axis('off')
                
                # 第二图：重建图像
                single_axes[1].imshow(recon_img_np)
                single_axes[1].set_title(f"reconstructed image (MSE: {mse:.4f})", fontsize=12)
                single_axes[1].axis('off')
                
                # 第三图：token索引热图 - 使用全局颜色映射
                single_axes[2].imshow(colored_tokens)
                single_axes[2].set_title("quantized token indices", fontsize=12)
                single_axes[2].set_xticks([])
                single_axes[2].set_yticks([])
                
                # 为热图上的不同token添加文本标签
                unique_tokens = np.unique(tk_labels_np)
                max_labels = len(unique_tokens)
                for idx, token_idx in enumerate(unique_tokens[:max_labels]):
                    # 找到该token的一个位置
                    positions = np.argwhere(tk_labels_np == token_idx)
                    if len(positions) > 0:
                        # 选择该token的中心位置
                        center_pos = positions[len(positions)//2]
                        y, x = center_pos
                        token_text = id2word.get(int(token_idx), f"UNK_{token_idx}")
                        # 添加文本标签，白色背景增加可读性
                        single_axes[2].text(x, y, token_text, color='black', fontsize=8,
                                         ha='center', va='center', 
                                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                
                # 第四图：文本描述
                single_axes[3].text(0.05, 0.5, sentence, 
                                  wrap=True, fontsize=10, va='center')
                single_axes[3].set_title("generated sentence", fontsize=12)
                single_axes[3].axis('off')
            
            plt.tight_layout()
            single_save_path = os.path.join(OUTPUT_DIR, f"image_{i+1}_{category.replace(' ', '_')}.png")
            plt.savefig(single_save_path, dpi=300, bbox_inches='tight')
            plt.close(single_fig)
            
            print(f"已保存图像 {i+1} 的可视化结果到 {single_save_path}")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 按类别分组创建对比图
    if group_by_category and len(samples_by_category) > 1:
        # 为每个类别创建一个对比图
        for category, samples in samples_by_category.items():
            if len(samples) < 2:
                continue  # 跳过只有一个样本的类别
                
            # 最多显示4个样本
            category_samples = samples[:min(4, len(samples))]
            n_samples = len(category_samples)
            
            # 创建一个大图，包含该类别的所有样本
            cat_fig, cat_axes = plt.subplots(n_samples, 3, figsize=(18, 5 * n_samples))
            
            # 如果只有一张图片，确保axes是二维的
            if n_samples == 1:
                cat_axes = np.array([cat_axes])
                
            for j, (cat_img, _) in enumerate(category_samples):
                # 预处理图片
                cat_img_tensor = preprocess_image(cat_img, device)
                
                # 获取量化token和重建图像
                with torch.no_grad():
                    cat_encoder_feature = model.quant_conv(model.encoder(cat_img_tensor))
                    cat_quant, cat_tk_labels, _ = model.quantize(cat_encoder_feature)
                    cat_reconstructed_img = model.decode(cat_quant)
                
                # 原始图像
                cat_img_np = cat_img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                cat_img_np = cat_img_np * 0.5 + 0.5
                cat_img_np = np.clip(cat_img_np, 0, 1)
                cat_axes[j, 0].imshow(cat_img_np)
                cat_axes[j, 0].set_title(f"sample {j+1}", fontsize=12)
                cat_axes[j, 0].axis('off')
                
                # 重建图像
                cat_recon_img_np = cat_reconstructed_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                cat_recon_img_np = cat_recon_img_np * 0.5 + 0.5
                cat_recon_img_np = np.clip(cat_recon_img_np, 0, 1)
                cat_axes[j, 1].imshow(cat_recon_img_np)
                
                # 计算MSE和PSNR
                cat_mse = np.mean((cat_img_np - cat_recon_img_np) ** 2)
                cat_psnr = 20 * np.log10(1.0 / np.sqrt(cat_mse)) if cat_mse > 0 else 100
                cat_axes[j, 1].set_title(f"reconstructed image (PSNR: {cat_psnr:.2f} dB)", fontsize=12)
                cat_axes[j, 1].axis('off')
                
                # Token索引热图 - 使用全局颜色映射
                cat_tk_labels_np = cat_tk_labels.squeeze().cpu().numpy()
                if cat_tk_labels_np.ndim == 1:
                    side_length = int(cat_tk_labels_np.shape[0] ** 0.5)
                    cat_tk_labels_np = cat_tk_labels_np.reshape(side_length, side_length)
                
                # 创建彩色token图
                height, width = cat_tk_labels_np.shape
                cat_colored_tokens = np.zeros((height, width, 4))
                for y in range(height):
                    for x in range(width):
                        token_id = cat_tk_labels_np[y, x]
                        cat_colored_tokens[y, x] = token_to_color.get(token_id, [0, 0, 0, 1])
                
                cat_axes[j, 2].imshow(cat_colored_tokens)
                cat_axes[j, 2].set_title("Token Index", fontsize=12)
                cat_axes[j, 2].set_xticks([])
                cat_axes[j, 2].set_yticks([])
                
                # 为热图上的不同token添加文本标签
                unique_tokens = np.unique(cat_tk_labels_np)
                max_labels = len(unique_tokens)
                for idx, token_idx in enumerate(unique_tokens[:max_labels]):
                    # 找到该token的一个位置
                    positions = np.argwhere(cat_tk_labels_np == token_idx)
                    if len(positions) > 0:
                        # 选择该token的中心位置
                        center_pos = positions[len(positions)//2]
                        y, x = center_pos
                        token_text = id2word.get(int(token_idx), f"UNK_{token_idx}")
                        # 添加文本标签，白色背景增加可读性
                        cat_axes[j, 2].text(x, y, token_text, color='black', fontsize=8,
                                         ha='center', va='center', 
                                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
            plt.suptitle(f"category: {category} - sample comparison", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.97])  # 为标题留出空间
            
            category_save_path = os.path.join(OUTPUT_DIR, f"category_{category.replace(' ', '_')}_comparison.png")
            plt.savefig(category_save_path, dpi=300, bbox_inches='tight')
            plt.close(cat_fig)
            
            print(f"已保存类别 '{category}' 的对比图到 {category_save_path}")
    
    # 创建一个图例图，显示常见token的颜色映射
    plt.figure(figsize=(12, 8))
    # 选择最常见的token（出现在多个图片中的）
    token_counts = {}
    for img, _ in samples_to_visualize:
        img_tensor = preprocess_image(img, device)
        with torch.no_grad():
            encoder_feature = model.quant_conv(model.encoder(img_tensor))
            _, tk_labels, _ = model.quantize(encoder_feature)
        tk_labels_np = tk_labels.squeeze().cpu().numpy()
        for token in np.unique(tk_labels_np):
            if token not in token_counts:
                token_counts[token] = 0
            token_counts[token] += 1
    
    # 选择出现频率最高的30个token
    top_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:100]
    
    # 创建图例
    for i, (token_id, count) in enumerate(top_tokens):
        plt.bar(i, count, color=token_to_color[token_id])
        token_text = id2word.get(int(token_id), f"UNK_{token_id}")
        plt.text(i, count/2, f"{token_text}\n({token_id})", ha='center', va='center', 
                rotation=90, fontsize=8, color='black',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    plt.title("Top 30 Most Common Tokens and Their Colors")
    plt.xlabel("Token")
    plt.ylabel("Frequency")
    plt.xticks([])
    plt.tight_layout()
    
    legend_save_path = os.path.join(OUTPUT_DIR, "token_color_legend.png")
    plt.savefig(legend_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存组合可视化结果到 {save_path}")
    print(f"已保存完整句子到 {txt_save_path}")
    print(f"已保存token颜色图例到 {legend_save_path}")
    
    return sentences

def visualize_category_similarity(category_token_usage, save_path='category_similarity.png'):
    """
    可视化不同类别之间的token使用相似度
    
    Args:
        category_token_usage: 字典，键为类别名称，值为Counter对象（token索引 -> 频率）
        save_path: 保存路径
    """
    # 确保保存路径在指定目录下
    save_path = os.path.join(OUTPUT_DIR, save_path)
    
    categories = list(category_token_usage.keys())
    n_categories = len(categories)
    
    # 创建相似度矩阵
    similarity_matrix = np.zeros((n_categories, n_categories))
    
    # 计算每对类别之间的余弦相似度
    for i, cat1 in enumerate(categories):
        counter1 = category_token_usage[cat1]
        vec1 = np.zeros(50000)  # 假设token索引不超过50000
        for idx, freq in counter1.items():
            if idx < 50000:
                vec1[idx] = freq
        
        # 归一化
        norm1 = np.linalg.norm(vec1)
        if norm1 > 0:
            vec1 = vec1 / norm1
        
        for j, cat2 in enumerate(categories):
            counter2 = category_token_usage[cat2]
            vec2 = np.zeros(50000)
            for idx, freq in counter2.items():
                if idx < 50000:
                    vec2[idx] = freq
            
            # 归一化
            norm2 = np.linalg.norm(vec2)
            if norm2 > 0:
                vec2 = vec2 / norm2
            
            # 计算余弦相似度
            similarity_matrix[i, j] = np.dot(vec1, vec2)
    
    # 可视化相似度矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='YlGnBu',
                xticklabels=categories, yticklabels=categories)
    plt.title('Category Similarity Based on Token Usage')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved category similarity visualization to {save_path}")
    
    # 使用层次聚类可视化类别关系
    plt.figure(figsize=(14, 8))
    linkage = sns.clustermap(similarity_matrix, 
                            method='average', 
                            cmap='YlGnBu',
                            xticklabels=categories,
                            yticklabels=categories,
                            figsize=(14, 12))
    plt.title('Hierarchical Clustering of Categories')
    cluster_save_path = save_path.replace('.png', '_cluster.png')
    plt.savefig(cluster_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved category clustering visualization to {cluster_save_path}")

def main():
    parser = get_args_parser()
    args = parser.parse_args()

    # ----------------------------------------------------------------------
    # Example usage: adjust your own paths below
    # ----------------------------------------------------------------------
    checkpoint_path = '/data/drstrange/yaxin/Projects/vision_llm/output_dir/vqvae_gpt2_codebook-256_mini_imagenet_improve/vqvae_checkpoint-200.pth'
    config_path     = '/data/drstrange/yaxin/Projects/vision_llm/vqgan_configs/v2l.yaml'

    # 使用datasets库加载mini-imagenet数据集样本
    data_samples = load_mini_imagenet_samples(
        num_samples=5,  # 每个类别加载5个样本
        split="train",  
        data_path="/data/drstrange/yaxin/data/mini-imagenet",  # 数据集路径
        max_categories=30  # 最多加载30个类别
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load VQVAE model
    vqvae_model = load_model(checkpoint_path, config_path, device, args)
    tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2-medium')

    # ----------------------------------------------------------------------
    # 1) Token Frequency by Category (使用VQVAE模型)
    # ----------------------------------------------------------------------
    print("\n=== 使用VQVAE模型分析token频率 ===")
    category_token_usage = analyze_token_frequency_per_category(
        model=vqvae_model, 
        tokenizer=tokenizer, 
        device=device, 
        data_samples=data_samples, 
        top_k=5
    )

    # 可视化token分布
    visualize_token_distribution_by_category(
        category_token_usage=category_token_usage,
        tokenizer=tokenizer,
        save_path='token_distribution_vqvae.png'
    )

    
    category_token_usage_llm = analyze_token_usage(
        vqvae_model=vqvae_model,
        tokenizer=tokenizer,
        device=device,
        data_samples=data_samples
        )
    
    # 可视化token分布
    visualize_token_distribution_by_category(
        category_token_usage=category_token_usage_llm,
        tokenizer=tokenizer,
        save_path='token_distribution_llm.png'
    )

    # 可视化类别相似度
    visualize_category_similarity(
        category_token_usage=category_token_usage,
        save_path='category_similarity_vqvae.png'
    )

    # ----------------------------------------------------------------------
    # 2) 增强可视化：多图像对比和重建
    # ----------------------------------------------------------------------
    print("\n=== 增强可视化：多图像对比和重建 ===")
    
    # 使用增强可视化功能
    visualize_multiple_images_with_tokens(
        model=vqvae_model,
        data_samples=data_samples,
        tokenizer=tokenizer,
        device=device,
        num_images=20,  # 增加可视化的图像数量
        save_path='enhanced_visualization.png',
        group_by_category=True,  # 按类别分组
        enhance_visualization=True  # 使用增强可视化
    )
    
    # 为了对比，也生成一个不分组的可视化
    visualize_multiple_images_with_tokens(
        model=vqvae_model,
        data_samples=data_samples[:10],  # 只使用前10个样本
        tokenizer=tokenizer,
        device=device,
        save_path='standard_visualization.png',
        group_by_category=False,  # 不按类别分组
        enhance_visualization=False  # 不使用增强可视化
    )

    sample_image, sample_cat = data_samples[-1]
    img_tensor = preprocess_image(sample_image, device)

    # 可视化单个图像的量化token
    visualize_quantized_tokens(
        model=vqvae_model,
        image_tensor=img_tensor,
        tokenizer=tokenizer,
        save_path=f'single_image_tokens_{sample_cat}.png'
    )

if __name__ == "__main__":
    main()
