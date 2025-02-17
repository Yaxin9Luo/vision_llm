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
from models.models_v2l import VQModel_LLaMA,VQModel_RoBERTa
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

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', help='whether to use distributed training')
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--imagenet_path", default="/root/autodl-tmp/data/imagenet", type=str, help="imagenet_path")
    parser.add_argument("--n_vision_words", default=8192, type=int)
    parser.add_argument("--n_class", default=10, type=int)    # CIFAR10有10个类别
    parser.add_argument("--vq_config_path", type=str, default="vqgan_configs/v2l.yaml", help="VQVAE config path")
    parser.add_argument("--image_size", type=int, default=256, help="input image size")
    parser.add_argument("--quantizer_type", type=str, default="org", help="quantizer type")
    parser.add_argument("--embed_dim", type=int, default=1024, help="embedding dimension")
    parser.add_argument("--rate_q", type=float, default=1, help="codebook loss weight")

    return parser

def main(args):

    misc.init_distributed_mode(args)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    if args.distributed:
        rank = dist.get_rank()
        print(f"Start running basic DDP example on rank {rank}.")
        device_id = rank % torch.cuda.device_count()
        print('I am rank {} using GPU {} on host {}'.format(rank,
        device_id, socket.gethostname()))
    else:
        rank = 0
        device_id = 0
    dataset_train = CIFAR10Dataset(
        partition="train", 
        device=device
    )
    dataset_val = CIFAR10Dataset(
        partition="val", 
        device=device
    )

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

    model = VQModel_RoBERTa(args=args, **config.model.params)
    model.to(device_id)
    # model.to(device)
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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id], find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    opt_ae = torch.optim.Adam(list(model_without_ddp.encoder.parameters())+
                            list(model_without_ddp.decoder.parameters())+
                            list(model_without_ddp.quant_conv.parameters())+
                            list(model_without_ddp.post_quant_conv.parameters()), lr=args.lr, betas=(0.5, 0.9), eps=1e-7)

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

