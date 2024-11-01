import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.models_v2l import VQModel_RoBERTa
import argparse
import os
from tqdm import tqdm
import yaml
from omegaconf import OmegaConf
from PIL import Image
import clip
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
class CIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, image_size=128):
        self.root = root
        self.train = train
        self.image_size = image_size

        _, self.clip_preprocessing = clip.load("ViT-L/14")

        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

        self.dataset = datasets.CIFAR10(root=self.root, train=self.train, download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = image.convert('RGB')

        clip_image = self.clip_preprocessing(image)

        input = torch.nn.functional.interpolate(clip_image.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False).squeeze(0)
        input = self.clip_std * input + self.clip_mean
        input = 2 * input - 1

        return input, clip_image, label
class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size]
        x = x.mean(dim=1)  # 或其他池化方式
        x = self.dropout(self.activation(self.linear1(x)))
        x = self.linear2(x)
        return x
def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
def load_cifar10(batch_size, image_size=128):
    train_dataset = CIFAR10Dataset(root='/root/autodl-tmp/data/', train=True, image_size=image_size)
    test_dataset = CIFAR10Dataset(root='/root/autodl-tmp/data/', train=False, image_size=image_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader


def train(model, classification_head, train_loader, optimizer, criterion, device):
    model.eval()  # Set VQModel to evaluation mode
    classification_head.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for input, _, target in progress_bar:
        input, target = input.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        with torch.no_grad():
            encoder_feature = model.quant_conv(model.encoder(input))
            encoder_feature = rearrange(encoder_feature, 'b c h w -> b h w c').contiguous()
            encoder_feature = encoder_feature.view(encoder_feature.size(0), -1, encoder_feature.size(-1))
            # _, _, llm_output = model(input, data_iter_step=0, step=0)
        # feature = llm_output[:, -1, :]
        output = classification_head(encoder_feature)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        progress_bar.set_postfix({'Loss': total_loss / (progress_bar.n + 1), 'Acc': 100. * correct / total})

    return total_loss / len(train_loader), 100. * correct / total

def test(model, classification_head, test_loader, criterion, device):
    model.eval()
    classification_head.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for input, clip_image, target in tqdm(test_loader, desc='Testing'):
            input, target = input.to(device), target.to(device)
            
            encoder_feature = model.quant_conv(model.encoder(input))
            encoder_feature = rearrange(encoder_feature, 'b c h w -> b h w c').contiguous()
            encoder_feature = encoder_feature.view(encoder_feature.size(0), -1, encoder_feature.size(-1))
            
            output = classification_head(encoder_feature)
            
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    return test_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Classification with VQModel_RoBERTa')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--model-path', type=str, required=True, help='path to the pre-trained VQModel_RoBERTa')
    parser.add_argument('--save-path', type=str, default='classification_head.pth', help='path to save the trained classification head')
    parser.add_argument("--vq_config_path", type=str, default="vqgan_configs/v2l.yaml", help="Decoding Loss")
    parser.add_argument("--stage", type=int, default=1, help="Decoding Loss")
    parser.add_argument("--embed_dim", type=int, default=1024, help="Decoding Loss")
    parser.add_argument("--quantizer_type", type=str, default="org", help="Decoding Loss")
    parser.add_argument("--rate_q", type=float, default=1, help="Decoding Loss")
    parser.add_argument("--rate_p", type=float, default=0.1, help="VGG Loss")
    parser.add_argument("--rate_d", type=float, default=0.75, help="GAN Loss")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # Load the configuration
    config = load_config(args.vq_config_path)
    print("Loaded config:")
    print(config)

    # Load your pre-trained VQModel_RoBERTa
    print("Initializing VQModel_RoBERTa...")
    vqmodel = VQModel_RoBERTa(args=args, **config.model.params)
    print("Model structure:")
    print(vqmodel)
    
    # 尝试加载模型
    print(f"Loading model from {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    print("Keys in loaded state dict:")
    print(state_dict.keys())
    
    if 'model' in state_dict:
        state_dict = state_dict['model']
        print("Found 'model' key, using its content")
        print("Keys in model state dict:")
        print(state_dict.keys())
    
    print("Keys in current model state dict:")
    print(vqmodel.state_dict().keys())
    
    print("Attempting to load state dict...")
    try:
        vqmodel.load_state_dict(state_dict, strict=False)
        print("State dict loaded successfully with strict=False")
    except Exception as e:
        print(f"Error loading state dict: {e}")
    
    vqmodel = vqmodel.to(device)
    vqmodel.eval()
    # Freeze the VQModel_RoBERTa
    for param in vqmodel.parameters():
        param.requires_grad = False
    total_params, trainable_params = count_parameters(vqmodel)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create classification head
    classification_head = MLPHead(1024,512,10).to(device)

    # Load CIFAR-10 data
    train_loader, test_loader = load_cifar10(args.batch_size)

    # Define optimizer and loss function
    optimizer = optim.Adam(classification_head.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0
    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f'Epoch {epoch}/{args.epochs}')
        train_loss, train_acc = train(vqmodel, classification_head, train_loader, optimizer, criterion, device)
        test_loss, test_acc = test(vqmodel, classification_head, test_loader, criterion, device)
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(classification_head.state_dict(), args.save_path)
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

    print(f"Training completed. Best accuracy: {best_accuracy:.2f}%")

if __name__ == '__main__':
    main()