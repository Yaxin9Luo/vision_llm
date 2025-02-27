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
    
    # 打印量化器配置
    print("\n量化器配置:")
    if 'quantizer' in config.model.params:
        print(yaml.dump(OmegaConf.to_container(config.model.params.quantizer)))
    else:
        print("未找到量化器配置")
    
    model = VQModel_LLaMA(args=args, **config.model.params)
    
    print(f"从 {checkpoint_path} 加载模型")
    # 添加weights_only=False来解决PyTorch 2.6的加载问题
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 获取模型状态字典
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("使用 'model' 键中的状态字典")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("使用 'state_dict' 键中的状态字典")
    else:
        print("未找到标准的模型权重键，尝试直接使用检查点")
        state_dict = checkpoint
    
    # 打印模型结构对比
    print("\n当前模型的键:")
    model_keys = set(model.state_dict().keys())
    for k in sorted(model_keys):
        print(f"- {k}")
    
    print("\n检查点中的键:")
    checkpoint_keys = set(state_dict.keys())

    
    # 检查键的差异
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys
    
    if missing_keys:
        print("\n缺失的键:")
        for k in sorted(missing_keys):
            print(f"- {k}")
    
    if unexpected_keys:
        print("\n未预期的键:")
        for k in sorted(unexpected_keys):
            print(f"- {k}")
    
    print("\n尝试加载状态字典...")
    try:
        # 首先尝试严格加载
        model.load_state_dict(state_dict)
        print("状态字典严格加载成功")
    except Exception as e1:
        print(f"严格加载失败: {e1}")
        print("尝试非严格加载...")
        try:
            # 如果严格加载失败，尝试非严格加载
            model.load_state_dict(state_dict, strict=False)
            print("状态字典非严格加载成功")
        except Exception as e2:
            print(f"非严格加载也失败了: {e2}")
            # 尝试处理键名不匹配的情况
            print("尝试处理键名不匹配...")
            new_state_dict = {}
            for k, v in state_dict.items():
                # 移除可能的'module.'前缀
                if k.startswith('module.'):
                    k = k[7:]
                # 移除可能的'model.'前缀
                if k.startswith('model.'):
                    k = k[6:]
                new_state_dict[k] = v
            
            try:
                model.load_state_dict(new_state_dict, strict=False)
                print("处理键名后非严格加载成功")
            except Exception as e3:
                print(f"所有加载尝试都失败了: {e3}")
                raise e3
    
    model = model.to(device)
    model.eval()
    return model

def get_image_from_parquet(parquet_path, index=0):
    """从parquet文件中读取图像"""
    # 读取parquet文件
    df = pd.read_parquet(parquet_path)
    
    # 获取图像数据
    image_data = df.iloc[index]['image']
    
    # 打印调试信息
    print(f"图像数据类型: {type(image_data)}")
    if isinstance(image_data, dict) and 'bytes' in image_data:
        # 如果是字典格式且包含bytes键
        image_bytes = image_data['bytes']
        image = Image.open(io.BytesIO(image_bytes))
    elif isinstance(image_data, dict) and 'array' in image_data:
        # 如果是字典格式且包含array键
        image_array = np.array(image_data['array'])
        image = Image.fromarray(image_array)
    elif isinstance(image_data, bytes):
        # 如果直接是字节格式
        image = Image.open(io.BytesIO(image_data))
    elif isinstance(image_data, np.ndarray):
        # 如果是numpy数组
        image = Image.fromarray(image_data)
    else:
        print(f"图像数据内容: {image_data}")
        raise ValueError(f"不支持的图像数据格式: {type(image_data)}")
    
    return image

def preprocess_image(image_path, is_parquet=True, parquet_index=0):
    """预处理图像，使用与训练时相同的CLIP预处理"""
    # 加载CLIP预处理器
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
    # 创建一个大的白色背景图片
    width = 1200
    height = 800
    background = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(background)
    
    # 调整输入图片大小
    image_size = 400
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    
    # 粘贴输入图片
    background.paste(image, (50, 50))
    
    # 设置字体（使用默认字体）
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # 添加标题
    draw.text((50, 20), "Input Image", fill='black', font=font)
    draw.text((50, 500), "Encoded Text:", fill='black', font=font)
    draw.text((50, 650), "Generated Text:", fill='black', font=font)
    
    # 包装文本以适应图片宽度
    wrapper = textwrap.TextWrapper(width=80)
    encoded_lines = wrapper.wrap(encoded_text)
    generated_lines = wrapper.wrap(generated_text)
    
    # 绘制编码文本
    y = 540
    for line in encoded_lines:
        draw.text((50, y), line, fill='blue', font=font)
        y += 25
    
    # 绘制生成文本
    y = 690
    for line in generated_lines:
        draw.text((50, y), line, fill='green', font=font)
        y += 25
    
    # 保存结果
    background.save(save_path)
    print(f"\n结果已保存到: {save_path}")

def get_token_sequence(model, image_tensor):
    """获取图像对应的token序列"""
    with torch.no_grad():
        # 获取编码特征
        encoder_feature = model.quant_conv(model.encoder(image_tensor))
        print(f"\n编码特征的形状: {encoder_feature.shape}")
        print(f"编码特征的值范围: [{encoder_feature.min().item():.4f}, {encoder_feature.max().item():.4f}]")
        
        # 量化得到tokens
        quant_out, token_indices, info = model.quantize(encoder_feature)
        print(f"\n量化后特征的形状: {quant_out.shape}")
        print(f"量化后特征的值范围: [{quant_out.min().item():.4f}, {quant_out.max().item():.4f}]")
        print(f"token_indices的形状: {token_indices.shape}")
        print(f"token_indices的值范围: [{token_indices.min().item()}, {token_indices.max().item()}]")
        print(f"量化信息: {info}")
        
        # 获取原始编码文本
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        # 检查tokenizer的词汇表
        print(f"\nTokenizer词汇表大小: {len(tokenizer)}")
        print(f"Token 11 对应的文本: {tokenizer.decode([11])}")
        
        # 尝试不同的解码方式
        token_list = token_indices.reshape(-1).tolist()
        print(f"\n前10个token: {token_list[:10]}")
        encoded_text = tokenizer.decode(token_indices.reshape(-1), skip_special_tokens=True)
        encoded_text_with_special = tokenizer.decode(token_indices.reshape(-1), skip_special_tokens=False)
        print("\n包含特殊token的解码结果:")
        print(encoded_text_with_special)
        
        return token_indices.reshape(1, -1), encoded_text

def generate_text(model, token_indices, max_new_tokens=100, temperature=0.7, top_p=0.9):
    """使用模型生成文本"""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    
    with torch.no_grad():
        # 使用GPT2模型生成后续token
        outputs = model.llm.generate(
            input_ids=token_indices,
            max_new_tokens=max_new_tokens,  # 使用max_new_tokens替代max_length
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        
        # 解码生成的token序列
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

def main():
    parser = argparse.ArgumentParser(description='Vision-LLM推理脚本')
    parser.add_argument('--config', type=str, 
                      default='/data/drstrange/yaxin/Projects/vision_llm/vqgan_configs/v2l.yaml',
                      help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, 
                      default='/data/drstrange/yaxin/Projects/vision_llm/output_dir/var_gpt2_mini_imagenet/var_checkpoint-last.pth',
                      help='模型检查点路径')
    parser.add_argument('--image', type=str, 
                      default='/data/drstrange/yaxin/data/mini-imagenet/data/validation-00000-of-00003.parquet',
                      help='输入图像路径')
    parser.add_argument('--parquet_index', type=int, default=0, help='parquet文件中的图像索引')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='生成的新token数量')
    parser.add_argument('--temperature', type=float, default=0.7, help='生成采样温度')
    parser.add_argument('--top_p', type=float, default=0.9, help='生成采样top_p值')
    parser.add_argument('--output', type=str, default='result.png', help='输出图片路径')
    
    args = parser.parse_args()
    # 添加模型所需的其他参数
    model_args = get_args_parser().parse_args([])
    for k, v in vars(model_args).items():
        if not hasattr(args, k):
            setattr(args, k, v)

    print(f"使用配置文件: {args.config}")
    print(f"使用检查点: {args.checkpoint}")
    print(f"使用parquet文件: {args.image}")
    print(f"使用图像索引: {args.parquet_index}")
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        model = load_model(args.checkpoint, args.config, device, args)
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return
    
    # 处理图像
    try:
        # 保存原始图像用于可视化
        original_image = get_image_from_parquet(args.image, args.parquet_index)
        image_tensor = preprocess_image(args.image, is_parquet=True, parquet_index=args.parquet_index).to(device)
        print("图像预处理完成")
    except Exception as e:
        print(f"图像处理失败: {str(e)}")
        return
    
    # 获取token序列和编码文本
    token_indices, encoded_text = get_token_sequence(model, image_tensor)
    print(f"\n图像编码得到token序列长度: {token_indices.shape[1]}")
    
    # 生成文本
    try:
        generated_text = generate_text(
            model, 
            token_indices, 
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        print("\n生成的文本:")
        print("-" * 50)
        print(generated_text)
        print("-" * 50)
        
        # 可视化结果
        visualize_results(original_image, encoded_text, generated_text, args.output)
        
    except Exception as e:
        print(f"文本生成失败: {str(e)}")
        return
    
    # 打印原始token序列
    print("\nToken序列:")
    print(token_indices.cpu().numpy())

if __name__ == '__main__':
    main() 