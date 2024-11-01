import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from models.models_v2l import VQModel_RoBERTa
from transformers import RobertaTokenizer
from visualization import load_model, preprocess_image, get_args_parser
import argparse

def get_image_embedding(model, image_tensor):
    with torch.no_grad():
        encoder_feature = model.quant_conv(model.encoder(image_tensor))
        _, tk_labels, _ = model.quantize(encoder_feature)
    return tk_labels.squeeze()
def get_text_embedding(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.forward(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()
def few_shot_classification(model, tokenizer, support_images, support_prompts, support_labels, query_image, query_prompt, k=1):
    support_embeddings = [torch.cat([get_image_embedding(model, img), get_text_embedding(model, tokenizer, prompt)]) for img, prompt in zip(support_images, support_prompts)]
    query_embedding = torch.cat([get_image_embedding(model, query_image), get_text_embedding(model, tokenizer, query_prompt)])
    
    similarities = []
    for embedding, label in zip(support_embeddings, support_labels):
        similarity = torch.cosine_similarity(query_embedding.float().unsqueeze(0), embedding.float().unsqueeze(0), dim=1)
        similarities.append((similarity.item(), label))
    
    similarities.sort(reverse=True)
    top_k = similarities[:k]
    
    return top_k

def visualize_results(support_images, support_labels, query_image, top_k):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Few-Shot Classification Results', fontsize=16)

    # Plot support images
    for i, (img, label) in enumerate(zip(support_images, support_labels)):
        row, col = divmod(i, 2)
        axs[row, col].imshow(img.squeeze().permute(1, 2, 0))
        axs[row, col].set_title(f"Support: {label}")
        axs[row, col].axis('off')

    # Plot query image
    axs[1, 1].imshow(query_image.squeeze().permute(1, 2, 0))
    axs[1, 1].set_title("Query Image")
    axs[1, 1].axis('off')

    # Plot results
    axs[1, 2].axis('off')
    axs[1, 2].text(0.1, 0.7, f"Prediction: {top_k[0][1]}", fontsize=12)
    axs[1, 2].text(0.1, 0.5, f"Confidence: {top_k[0][0]:.4f}", fontsize=12)

    plt.tight_layout()
    plt.savefig('few_shot_classification_results.png')
    plt.close()

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    checkpoint_path = '/root/autodl-tmp/vision_llm/output_dir/frozen_roberta_codebook/vqgan_checkpoint-last.pth'
    config_path = '/root/autodl-tmp/vision_llm/vqgan_configs/v2l.yaml'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_model(checkpoint_path, config_path, device, args)
    tokenizer = RobertaTokenizer.from_pretrained('/root/autodl-tmp/roberta-large')
    # Load and preprocess support images
    support_image_paths = [
        '/root/autodl-tmp/ImageNet/val/n01440764/ILSVRC2012_val_00023559.JPEG',
        '/root/autodl-tmp/ImageNet/val/n02094433/ILSVRC2012_val_00000949.JPEG',
        '/root/autodl-tmp/ImageNet/val/n01807496/ILSVRC2012_val_00001563.JPEG',
        '/root/autodl-tmp/ImageNet/val/n02120079/ILSVRC2012_val_00002447.JPEG'
    ]
    support_images = [preprocess_image(path, device) for path in support_image_paths]
    support_labels = ['fish', 'dog', 'bird', 'fox']
    support_prompts = [
        "This is a fish.",
        "This is a dog.",
        "This is a bird.",
        "This is a fox."
    ]
    # Load and preprocess query image
    query_image_path = '/root/autodl-tmp/ImageNet/val/n02094433/ILSVRC2012_val_00045790.JPEG'
    query_image = preprocess_image(query_image_path, device)
    query_prompt = "What animal is this?"
    
    top_k = few_shot_classification(model, tokenizer, support_images, support_prompts, support_labels, query_image, query_prompt)
    
    print("Prediction:")
    for score, label in top_k:
        print(f"{label}: {score:.4f}")
    
    visualize_results(support_images, support_labels, query_image, top_k)
    print("Visualization saved as 'few_shot_classification_results.png'")

if __name__ == "__main__":
    main()