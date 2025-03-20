import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from models.var_models import VQModel_LLaMA
from omegaconf import OmegaConf
import argparse
from transformers import GPT2Tokenizer
import clip
import numpy as np
import yaml
import pandas as pd
import io
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import textwrap
from einops import rearrange

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--n_vision_words", default=8192, type=int)
    parser.add_argument("--output_type", default="next_token_prediction", type=str)
    parser.add_argument("--decode_rate", type=float, default=0)
    parser.add_argument("--n_class", default=1000, type=int)    
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
    """加载模型和配置"""
    config = load_config(config_path, display=True)
    print("初始化 VQModel_LLaMA...")
    
    model = VQModel_LLaMA(args=args, **config.model.params)
    
    print(f"从 {checkpoint_path} 加载模型")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 获取模型状态字典
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    try:
        # 尝试加载状态字典
        model.load_state_dict(state_dict, strict=False)
        print("状态字典加载成功")
    except Exception as e:
        print(f"加载失败: {e}")
        # 处理键名不匹配的情况
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            if k.startswith('model.'):
                k = k[6:]
            new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print("处理键名后加载成功")
    
    model = model.to(device)
    model.eval()
    return model

def get_image_from_parquet(parquet_path, index=0):
    """从parquet文件中读取图像"""
    df = pd.read_parquet(parquet_path)
    image_data = df.iloc[index]['image']
    
    if isinstance(image_data, dict) and 'bytes' in image_data:
        image_bytes = image_data['bytes']
        image = Image.open(io.BytesIO(image_bytes))
    elif isinstance(image_data, dict) and 'array' in image_data:
        image_array = np.array(image_data['array'])
        image = Image.fromarray(image_array)
    elif isinstance(image_data, bytes):
        image = Image.open(io.BytesIO(image_data))
    elif isinstance(image_data, np.ndarray):
        image = Image.fromarray(image_data)
    else:
        raise ValueError(f"不支持的图像数据格式: {type(image_data)}")
    
    return image

def preprocess_image(image_path, is_parquet=True, parquet_index=0):
    """预处理图像，使用与训练时相同的CLIP预处理"""
    _, clip_preprocessing = clip.load("ViT-L/14")
    
    # CLIP归一化参数
    clip_mean = torch.from_numpy(np.array([0.48145466, 0.4578275, 0.40821073])).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).float()
    clip_std = torch.from_numpy(np.array([0.26862954, 0.26130258, 0.27577711])).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).float()
    
    # 加载图像
    if is_parquet:
        image = get_image_from_parquet(image_path, parquet_index)
    else:
        image = Image.open(image_path).convert('RGB')
    
    # 应用CLIP预处理
    clip_image = clip_preprocessing(image)
    
    # 处理输入图像
    input = torch.nn.functional.interpolate(
        clip_image.unsqueeze(0), 
        size=(128, 128), 
        mode='bilinear', 
        align_corners=False
    ).contiguous()
    
    input = clip_std * input + clip_mean
    input = 2 * input - 1
    
    return input

def visualize_results(image, encoded_text, generated_text, save_path='result.png'):
    """将图片和文本结果可视化并保存"""
    # 增加画布高度以容纳更多文本
    width = 1200
    height = 1600
    background = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(background)
    
    # 增加输入图片大小
    image_size = 500
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    
    # 粘贴输入图片
    background.paste(image, (50, 50))
    
    # 设置字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # 添加标题
    draw.text((50, 20), "Input Image", fill='black', font=font)
    draw.text((50, 580), "Encoded Text:", fill='black', font=font)
    # 增加文本间距
    draw.text((50, 850), "Generated Text:", fill='black', font=font)
    
    # 包装文本以适应图片宽度
    wrapper = textwrap.TextWrapper(width=80)
    encoded_lines = wrapper.wrap(encoded_text)
    generated_lines = wrapper.wrap(generated_text)
    
    # 绘制编码文本
    y = 620
    for line in encoded_lines:
        draw.text((50, y), line, fill='blue', font=font)
        y += 25
    
    # 绘制生成文本
    y = 890
    for line in generated_lines:
        draw.text((50, y), line, fill='green', font=font)
        y += 25
    
    # 保存结果
    background.save(save_path)
    print(f"\n结果已保存到: {save_path}")

def map_embeddings_to_tokens(model, embeddings):
    """将嵌入映射到最近的codebook token"""
    with torch.no_grad():
        # 重塑嵌入以便于处理
        orig_shape = embeddings.shape
        if len(orig_shape) == 3:  # [B, L, C]
            embeddings_flat = embeddings.reshape(-1, embeddings.shape[-1])
        elif len(orig_shape) == 4:  # [B, C, H, W]
            embeddings_flat = rearrange(embeddings, 'b c h w -> (b h w) c').contiguous()
        else:
            raise ValueError(f"不支持的嵌入形状: {orig_shape}")
        
        # 计算与codebook的距离
        d = torch.sum(embeddings_flat ** 2, dim=1, keepdim=True) + \
           torch.sum(model.codebook.detach()**2, dim=1) - 2 * \
           torch.einsum('bd,dn->bn', embeddings_flat, rearrange(model.codebook.detach(), 'n d -> d n'))
        
        # 找到最近的codebook索引
        token_indices = torch.argmin(d, dim=1)
        
        # 重塑回原始形状
        if len(orig_shape) == 3:  # [B, L, C]
            token_indices = token_indices.reshape(orig_shape[0], orig_shape[1])
        elif len(orig_shape) == 4:  # [B, C, H, W]
            token_indices = token_indices.reshape(orig_shape[0], orig_shape[2], orig_shape[3])
            
        return token_indices

def direct_quantize(model, image_tensor):
    """直接量化图像为token序列和文本"""
    with torch.no_grad():
        # 获取编码特征
        encoder_feature = model.quant_conv(model.encoder(image_tensor))
        
        # 量化得到tokens
        quant_out, token_indices, codebook_loss, unique_tokens, unique_ratio = model.quantize(encoder_feature)
        
        # 重塑token索引为序列
        token_indices = token_indices.reshape(1, -1)
        
        # 解码为文本
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        encoded_text = tokenizer.decode(token_indices.reshape(-1), skip_special_tokens=True)
        
        print(f"\nImage Encoded Text:")
        print(f"unique tokens: {unique_tokens}")
        print(f"sequence length: {token_indices.shape[1]}")
        
        return token_indices, encoded_text

def generate_text_with_predictor(model, token_indices, max_new_tokens=100, temperature=0.7):
    """使用模型的next_patch_predictor进行自回归生成"""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    
    with torch.no_grad():
        device = token_indices.device
        
        # 将token_indices转换为嵌入
        if len(token_indices.shape) == 1:
            token_indices = token_indices.unsqueeze(0)
            
        # 获取嵌入
        token_embeddings = F.embedding(token_indices, model.codebook)
        
        # 准备自回归序列
        bos_embedding = model.bos_embedding.expand(token_indices.shape[0], -1, -1)
        
        # 创建输入序列 [BOS, token_embeddings]
        input_seq = torch.cat([bos_embedding, token_embeddings], dim=1)
        
        # 生成max_new_tokens个新token
        generated_tokens = []
        for _ in range(max_new_tokens):
            # 获取位置ID
            position_ids = model.position_ids[:, :input_seq.shape[1]]
            
            # 通过GPT2进行自回归预测
            llm_output = model.llm(inputs_embeds=input_seq, position_ids=position_ids).last_hidden_state
            
            # 预测下一个patch嵌入
            next_embedding = model.next_patch_predictor(llm_output[:, -1:])
            
            # 添加噪声以增加多样性
            if temperature > 0:
                noise = torch.randn_like(next_embedding) * temperature
                next_embedding = next_embedding + noise
            
            # 将预测的嵌入映射到最近的codebook token
            next_token_indices = map_embeddings_to_tokens(model, next_embedding)
            next_token = next_token_indices.reshape(-1)[0].item()
            generated_tokens.append(next_token)
            
            # 将token转换回嵌入
            next_token_embedding = F.embedding(torch.tensor([next_token], device=device), model.codebook).unsqueeze(0)
            
            # 添加到输入序列
            input_seq = torch.cat([input_seq, next_token_embedding], dim=1)
            
            # 如果生成了EOS token，停止生成
            if next_token == tokenizer.eos_token_id:
                break
        
        # 只解码生成的token序列，不包含输入序列
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"生成文本长度: {len(generated_text)}")
        print(f"生成token数量: {len(generated_tokens)}")
        return generated_text, generated_tokens

def main():
    parser = argparse.ArgumentParser(description='Vision-LLM推理脚本')
    parser.add_argument('--config', type=str, 
                      default='/data/drstrange/yaxin/Projects/vision_llm/vqgan_configs/v2l.yaml',
                      help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, 
                      default='/data/drstrange/yaxin/Projects/vision_llm/output_dir/var_train_scratch_gpt2-medium_mini_imagenet-no-recon/var_checkpoint-last.pth',
                      help='模型检查点路径')
    parser.add_argument('--image', type=str, 
                      default='/data/drstrange/yaxin/data/mini-imagenet/data/validation-00000-of-00003.parquet',
                      help='输入图像路径')
    parser.add_argument('--parquet_index', type=int, default=800, help='parquet文件中的图像索引')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='生成的新token数量')
    parser.add_argument('--temperature', type=float, default=0.7, help='生成采样温度')
    
    args = parser.parse_args()
    # 添加模型所需的其他参数
    model_args = get_args_parser().parse_args([])
    for k, v in vars(model_args).items():
        if not hasattr(args, k):
            setattr(args, k, v)

    print(f"使用配置文件: {args.config}")
    print(f"使用检查点: {args.checkpoint}")
    print(f"使用图像文件: {args.image}")
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = load_model(args.checkpoint, args.config, device, args)
    print("模型加载成功")
    
    # 处理图像
    original_image = get_image_from_parquet(args.image, args.parquet_index)
    image_tensor = preprocess_image(args.image, is_parquet=True, parquet_index=args.parquet_index).to(device)
    
    token_indices, direct_text = direct_quantize(model, image_tensor)
    
    generated_text, generated_tokens = generate_text_with_predictor(
        model, 
        token_indices, 
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    output_path = f"result_{args.parquet_index}.png"
    # 可视化结果
    visualize_results(original_image, direct_text, generated_text, output_path)
    print(f"结果已保存到: {output_path}")

if __name__ == '__main__':
    main() 