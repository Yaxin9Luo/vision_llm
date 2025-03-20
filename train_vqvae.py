import argparse
import datetime
import json
import os
import time
from pathlib import Path
import albumentations
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import yaml
import torch
import clip
from omegaconf import OmegaConf
import torch.distributed as dist
from models.models_v2l import *
from training.engine_training import train_one_epoch
import util.misc as misc
import socket
from torchvision import datasets, transforms
from util.misc import NativeScalerWithGradNormCount as NativeScaler

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):

  model = VQModel_LLaMA(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

class TinyImageNetDataset(Dataset):
    def __init__(self, data_root, image_size=64, max_words=30, n_class=200, partition="train", device="cpu"):
        self.max_words = max_words
        self.device = device
        self.image_size = image_size

        self.data_root = data_root
        _, self.clip_preprocessing = clip.load("ViT-L/14")

        self.rescaler = albumentations.SmallestMaxSize(max_size=64)
        self.cropper = albumentations.CenterCrop(height=64, width=64)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

        self.token_nums = [1, 4, 16, 64, 256, 256]
        self.partition = partition

        self.clip_mean = torch.from_numpy(np.array([0.48145466, 0.4578275, 0.40821073])).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).float()
        self.clip_std = torch.from_numpy(np.array([0.26862954, 0.26130258, 0.27577711])).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).float()

        self.image_ids = []
        self.class_labels = []
        
        # 加载类别映射 - 移到这里，让所有分支都能访问
        wnids_path = os.path.join(self.data_root, "wnids.txt")
        with open(wnids_path, "r") as f:
            wnids = [x.strip() for x in f.readlines()]
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
        
        # 处理不同的数据集划分
        if partition == "train":
            partition_dir = os.path.join(self.data_root, "train")
            class_dirs = [d for d in os.listdir(partition_dir) if os.path.isdir(os.path.join(partition_dir, d))]
            
            for class_dir in class_dirs:
                if class_dir in self.class_to_idx:
                    class_label = self.class_to_idx[class_dir]
                    images_dir = os.path.join(partition_dir, class_dir)
                    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.JPEG', '.jpg', '.png'))]
                    
                    for image_file in image_files:
                        self.image_ids.append(os.path.join(class_dir, image_file))
                        self.class_labels.append(class_label)
                        
        elif partition == "val":
            val_dir = os.path.join(self.data_root, "val")
            # 加载验证集标注
            val_annotations_path = os.path.join(val_dir, "val_annotations.txt")
            with open(val_annotations_path, "r") as f:
                for line in f:
                    image_file, class_dir, *_ = line.strip().split("\t")
                    self.image_ids.append(os.path.join("images", image_file))
                    self.class_labels.append(self.class_to_idx[class_dir])

        print(f"Using {len(self.image_ids)} images for {partition} set")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        try:
            image_id = self.image_ids[index]
            if self.partition == "train":
                image_path = os.path.join(self.data_root, "train", image_id)
            else:
                image_path = os.path.join(self.data_root, "val", image_id)
                
            image = Image.open(image_path).convert('RGB')
            clip_image = self.clip_preprocessing(image)
            label = self.class_labels[index]

            input = torch.nn.functional.interpolate(clip_image.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).contiguous()
            input = self.clip_std * input + self.clip_mean
            input = 2 * input - 1
            input = input.squeeze(0)

            return [image_id, input, clip_image, label]
        except Exception as e:
            print(f"Error processing index {index}: {str(e)}")
            return self.__getitem__((index + 1) % len(self))


class CIFAR10Dataset(Dataset):
    def __init__(self, partition="train", device="cpu"):
        self.device = device
        self.partition = partition
        
        # CLIP preprocessing
        _, self.clip_preprocessing = clip.load("ViT-L/14", device='cpu')  # 强制在CPU上加载CLIP
        
        # CLIP normalization parameters
        self.clip_mean = torch.from_numpy(np.array([0.48145466, 0.4578275, 0.40821073])).float()
        self.clip_std = torch.from_numpy(np.array([0.26862954, 0.26130258, 0.27577711])).float()
        
        # Load CIFAR10
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.dataset = datasets.CIFAR10(
            root='/home/yaxin/data', 
            train=(partition == "train"),
            download=True, 
            transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            image, label = self.dataset[index]
            image = Image.fromarray(np.uint8(image.permute(1,2,0).numpy() * 255))
            
            # Apply CLIP preprocessing
            with torch.no_grad():  # 减少内存使用
                clip_image = self.clip_preprocessing(image)
            
            # Process input for VQVAE
            with torch.cuda.amp.autocast():  # 使用混合精度
                input = torch.nn.functional.interpolate(
                    clip_image.unsqueeze(0), 
                    size=(128, 128), 
                    mode='bilinear', 
                    align_corners=False
                ).contiguous()
                input = self.clip_std.to(input.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * input + \
                       self.clip_mean.to(input.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                input = 2 * input - 1
                input = input.squeeze(0)
            
            return [str(index), input, clip_image, label]
        except Exception as e:
            print(f"Error processing index {index}: {str(e)}")
            # 返回一个替代样本
            return self.__getitem__((index + 1) % len(self))
class ImageNetDataset(Dataset):
    def __init__(self, data_root, image_size, max_words=30, n_class=1000, partition="train", device="cpu", use_subset=1.0):
        self.max_words = max_words
        self.device = device
        self.image_size = image_size

        self.data_root = data_root
        _, self.clip_preprocessing = clip.load("ViT-L/14")

        self.rescaler = albumentations.SmallestMaxSize(max_size=128)
        self.cropper = albumentations.RandomCrop(height=128, width=128)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

        self.token_nums = [1, 4, 16, 64, 256, 256]

        self.partition = partition

        self.clip_mean = torch.from_numpy(np.array([0.48145466, 0.4578275, 0.40821073])).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).float()
        self.clip_std = torch.from_numpy(np.array([0.26862954, 0.26130258, 0.27577711])).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).float()

        self.image_ids = []
        self.class_labels = []

        partition_dir = os.path.join(self.data_root, partition)
        class_dirs = [d for d in os.listdir(partition_dir) if os.path.isdir(os.path.join(partition_dir, d))]
        
        for class_label, class_dir in enumerate(class_dirs):
            if class_label >= n_class:
                break
            class_path = os.path.join(partition_dir, class_dir)
            image_files = [f for f in os.listdir(class_path) if f.endswith(('.JPEG', '.jpg', '.png'))]
            subset_size = max(1, int(len(image_files) * use_subset))
            selected_images = np.random.choice(image_files, subset_size, replace=False)
            
            for image_file in selected_images:
                self.image_ids.append(os.path.join(class_dir, image_file))
                self.class_labels.append(class_label)

        print(f"Using {len(self.image_ids)} images for {partition} set")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_path = os.path.join(self.data_root, self.partition, image_id)
        image = Image.open(image_path).convert('RGB')
        clip_image = self.clip_preprocessing(image)
        label = self.class_labels[index]

        input = torch.nn.functional.interpolate(clip_image.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False).contiguous()
        input = self.clip_std * input + self.clip_mean
        input = 2 * input - 1
        input = input.squeeze(0)

        return [image_id, input, clip_image, label]
class MiniImageNetDataset(Dataset):
    def __init__(self, max_words=30, image_size=256,n_class=1000, partition="train", device="cpu"):
        self.max_words = max_words
        self.device = device
        self.image_size = image_size
        self.n_class = n_class
        # 加载CLIP模型进行预处理
        _, self.clip_preprocessing = clip.load("ViT-L/14")
        self.rescaler = albumentations.SmallestMaxSize(max_size=128)
        self.cropper = albumentations.RandomCrop(height=128, width=128)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

        self.token_nums = [1, 4, 16, 64, 256, 256]
        # CLIP归一化参数
        self.clip_mean = torch.from_numpy(np.array([0.48145466, 0.4578275, 0.40821073])).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).float()
        self.clip_std = torch.from_numpy(np.array([0.26862954, 0.26130258, 0.27577711])).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).float()
        
        # 从Hugging Face加载Mini-ImageNet数据集
        from datasets import load_dataset
        self.dataset = load_dataset("/data/drstrange/yaxin/data/mini-imagenet", split=partition)
        
        print(f"Loaded {len(self.dataset)} images for {partition} set")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            sample = self.dataset[index]
            image = sample['image']
            label = sample['label']
            
            # 将PIL Image转换为RGB模式
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            image = image.convert('RGB')
            
            # 应用CLIP预处理
            clip_image = self.clip_preprocessing(image)
            
            # 处理输入图像
            input = torch.nn.functional.interpolate(
                clip_image.unsqueeze(0), 
                size=(128, 128), 
                mode='bilinear', 
                align_corners=False
            ).contiguous()
            
            input = self.clip_std * input + self.clip_mean
            input = 2 * input - 1
            input = input.squeeze(0)
            
            return [str(index), input, clip_image, label]
            
        except Exception as e:
            print(f"Error processing index {index}: {str(e)}")
            return self.__getitem__((index + 1) % len(self))

def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )
    parser.add_argument("--stage", type=int, default=1, help="Pretraining stage")
    parser.add_argument("--max_seq_len", type=int, default=512, metavar="LENGTH", help="the maximum sequence length")
    parser.add_argument("--use_mod", type=bool, default=True, help="use MoD for masking")
    parser.add_argument("--capacity_factor", type=float, default=0.5, help="capacity factor for MoD")
    parser.add_argument("--router_aux_loss_coef", type=float, default=0.01, help="auxiliary loss coefficient for router")
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
    parser.add_argument("--output_dir", default="./output_dir/vqvae_roberta_mae", help="path where to save")
    parser.add_argument("--log_dir", default="./output_dir/vqvae_roberta_mae", help="path where to tensorboard log")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--dataset", type=str, default="imagenet", help="dataset name")
    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', help='whether to use distributed training')
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--dist_backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--data_path", default="/root/autodl-tmp/data/imagenet", type=str, help="imagenet_path")
    parser.add_argument("--n_vision_words", default=8192, type=int)
    parser.add_argument("--n_class", default=10, type=int)    # CIFAR10有10个类别
    parser.add_argument("--vq_config_path", type=str, default="vqgan_configs/v2l.yaml", help="VQVAE config path")
    parser.add_argument("--image_size", type=int, default=256, help="input image size")
    parser.add_argument("--quantizer_type", type=str, default="org", help="quantizer type")
    parser.add_argument("--embed_dim", type=int, default=1024, help="embedding dimension")
    parser.add_argument("--rate_q", type=float, default=1, help="codebook loss weight")

    return parser

def main(args):
    # 在init_distributed_mode之前设置cuda设备
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.distributed = True
        torch.cuda.set_device(args.gpu)
        print(f'Using GPU: {args.gpu} for training')

    misc.init_distributed_mode(args)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    # 更新device设置
    if args.distributed:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # 确保在数据加载之前进行同步
    if args.distributed:
        torch.distributed.barrier(device_ids=[args.gpu])

    if args.dataset == "cifar10":
        dataset_train = CIFAR10Dataset(
            partition="train", 
            device=device
        )
        dataset_val = CIFAR10Dataset(
            partition="val", 
            device=device
        )
    elif args.dataset == "imagenet":
        dataset_train = ImageNetDataset(
            data_root=args.data_path, image_size=args.image_size, max_words=args.max_seq_len, n_class=args.n_class, partition="train", device=device
        )
        dataset_val = ImageNetDataset(
            data_root=args.data_path, image_size=args.image_size, max_words=args.max_seq_len, n_class=args.n_class, partition="val", device=device
        )
    elif args.dataset == "tiny-imagenet":
        dataset_train = TinyImageNetDataset(
            data_root=args.data_path, image_size=args.image_size, max_words=args.max_seq_len, n_class=args.n_class, partition="train", device=device
        )
        dataset_val = TinyImageNetDataset(
            data_root=args.data_path, image_size=args.image_size, max_words=args.max_seq_len, n_class=args.n_class, partition="val", device=device
        )
    elif args.dataset == "mini-imagenet":
        dataset_train = MiniImageNetDataset(
           image_size=args.image_size, max_words=args.max_seq_len, n_class=args.n_class, partition="train", device=device
        )
        dataset_val = MiniImageNetDataset(
        image_size=args.image_size, max_words=args.max_seq_len, n_class=args.n_class, partition="validation", device=device
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, drop_last=True
        )

        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True, drop_last=False
        )

        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )


    config = load_config(args.vq_config_path, display=True)

    model = VQVAE_LLM_Codebook(args=args, **config.model.params)
    if args.distributed:
        model.to(device)  # 使用更新后的device
    else:
        model.to(torch.device(args.device))
    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                         device_ids=[args.gpu],
                                                         output_device=args.gpu,
                                                         find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    opt_ae = torch.optim.Adam(list(model_without_ddp.encoder.parameters())+
                            list(model_without_ddp.decoder.parameters())+
                            list(model_without_ddp.quant_conv.parameters())+
                            list(model_without_ddp.post_quant_conv.parameters())+
                            list(model_without_ddp.adaptor.parameters()), lr=args.lr, betas=(0.5, 0.9), eps=1e-7)

    loss_scaler_ae = NativeScaler()

    ##auto resume
    if os.path.exists(os.path.join(args.output_dir, 'vqvae_checkpoint-last.pth')):
        ckpt = torch.load(os.path.join(args.output_dir, 'vqvae_checkpoint-last.pth'), 
                        map_location="cpu",
                        weights_only=False)
        model_without_ddp.load_state_dict(ckpt["model"], strict=True)
        opt_ae.load_state_dict(ckpt["optimizer"])
        loss_scaler_ae.load_state_dict(ckpt["scaler"])
        args = ckpt["args"]
        args.start_epoch = ckpt["epoch"] + 1
        print(args)
        print("*********Resuming From Epoch %d********"%(args.start_epoch))

    optimizer = opt_ae
    loss_scaler = loss_scaler_ae

    num_val_images = len(dataset_val)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            data_loader_val.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args
        )

        misc.save_model_last_vqvae(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
        )
        
        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            misc.save_model_vqvae(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch#,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

