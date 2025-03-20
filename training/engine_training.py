from typing import Iterable
from tqdm import tqdm  
import torch
import util.lr_sched as lr_sched
import util.misc as misc
import mlflow
from torch.nn import functional as F
import pytorch_lightning as pl
import importlib
from einops import rearrange
from torch.nn import Embedding
import torch.nn as nn
import numpy as np
import torchvision

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    opt_ae = optimizer
    loss_scaler_ae = loss_scaler
    
    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))
        
    # Get the actual model (handle DDP wrapped case)
    actual_model = model.module if hasattr(model, 'module') else model
        
    for data_iter_step, [image_ids, images, clip_images, label_cls] in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        cur_iter = len(data_loader) * epoch + data_iter_step

        ####Tokenizer with VQ-VAE
        b = images.shape[0]
        x = images.to(device)
        label_cls = label_cls.to(device)

        loss, rec_loss, codebook_loss, tk_labels, dec, unique_tokens_in_batch = model(x, cur_iter, step=0)
        
        # 记录unique tokens相关的指标
        metric_logger.update(unique_tokens_in_batch=unique_tokens_in_batch)
        
        # MoD aux_loss
        if hasattr(actual_model, 'use_mod') and actual_model.use_mod and actual_model.training:
            aux_loss = actual_model.last_aux_loss  # Need to save the last aux_loss in the model
            aux_loss_value = aux_loss.item()
            metric_logger.update(aux_loss=aux_loss_value)
            misc.all_reduce_mean(aux_loss_value)
            aux_loss_value_reduce = misc.all_reduce_mean(aux_loss_value)
        
        opt_ae.zero_grad() 

        lr_sched.adjust_learning_rate(opt_ae, data_iter_step / len(data_loader) + epoch, args)

        loss_scaler_ae(loss, opt_ae, parameters=list(actual_model.encoder.parameters())+
                                    list(actual_model.decoder.parameters())+
                                    list(actual_model.quant_conv.parameters())+
                                    list(actual_model.post_quant_conv.parameters())+
                                    list(actual_model.adaptor.parameters()), 
                                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        torch.cuda.synchronize()
        
        lr = opt_ae.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        loss_value = loss.item()
        metric_logger.update(loss=loss_value)
        misc.all_reduce_mean(loss_value)
        loss_value_reduce = misc.all_reduce_mean(loss_value)

        recloss_value = rec_loss.item()
        metric_logger.update(recloss=recloss_value)
        misc.all_reduce_mean(recloss_value)
        recloss_value_reduce = misc.all_reduce_mean(recloss_value)

        codebook_loss_value = codebook_loss.item()
        metric_logger.update(codebook_loss=codebook_loss_value)
        misc.all_reduce_mean(codebook_loss_value)
        codebook_loss_value_reduce = misc.all_reduce_mean(codebook_loss_value)

        if log_writer is not None and cur_iter % 1000 == 0:
            epoch_1000x = int(cur_iter)
            log_writer.add_scalar("Iter/lr", lr, epoch_1000x)
            log_writer.add_scalar("Iter/Loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("Iter/REC Loss", recloss_value_reduce, epoch_1000x)
            log_writer.add_scalar("Iter/Codebook Loss", codebook_loss_value_reduce, epoch_1000x)
            # 添加unique tokens的日志
            log_writer.add_scalar("Iter/Unique Tokens in Batch", unique_tokens_in_batch, epoch_1000x)

            
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if log_writer is not None:
        log_writer.add_scalar("Epoch/Loss", loss_value_reduce, epoch)
        log_writer.add_scalar("Epoch/REC Loss", recloss_value_reduce, epoch)
        log_writer.add_scalar("Epoch/Codebook Loss", codebook_loss_value_reduce, epoch)
        log_writer.add_scalar("Epoch/Unique Tokens in Batch", unique_tokens_in_batch, epoch)
            
        save_x = (x-x.min())/(x.max()-x.min())
        save_xrec = (dec-dec.min())/(dec.max()-dec.min())
        save_img = torch.cat([save_x, save_xrec], dim=-1).detach().cpu().numpy()
        for b in range(0, save_img.shape[0]):
            mlflow.log_image(save_img[b].transpose(1, 2, 0), "recons_%s_%s.png"%(epoch, b))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_var(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    opt_ae = optimizer
    loss_scaler_ae = loss_scaler
    
    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))
        
    # Get the actual model (handle DDP wrapped case)
    actual_model = model.module if hasattr(model, 'module') else model
        
    for data_iter_step, [image_ids, images, clip_images, label_cls] in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        cur_iter = len(data_loader) * epoch + data_iter_step

        b = images.shape[0]
        x = images.to(device)
        label_cls = label_cls.to(device)

        # Use the modified forward function
        loss, tk_labels = model(x, cur_iter, step=0)

        
        opt_ae.zero_grad() 

        lr_sched.adjust_learning_rate(opt_ae, data_iter_step / len(data_loader) + epoch, args)

        # Update parameters
        loss_scaler_ae(loss, opt_ae, parameters=list(actual_model.llm.parameters())+
                                        list(actual_model.adaptor.parameters()),  
                                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        torch.cuda.synchronize()
        
        lr = opt_ae.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        loss_value = loss.item()
        metric_logger.update(loss=loss_value)
        misc.all_reduce_mean(loss_value)
        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None and cur_iter % 1000 == 0:
            epoch_1000x = int(cur_iter)
            log_writer.add_scalar("Iter/lr", lr, epoch_1000x)
            log_writer.add_scalar("Iter/AR Loss", loss_value_reduce, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if log_writer is not None:
        log_writer.add_scalar("Epoch/AR Loss", loss_value_reduce, epoch)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}