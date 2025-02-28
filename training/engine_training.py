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

        loss, rec_loss, codebook_loss, tk_labels, dec = model(x, cur_iter, step=0)
        
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
                                    list(actual_model.post_quant_conv.parameters()), 
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
            if hasattr(actual_model, 'use_mod') and actual_model.use_mod and actual_model.training:
                log_writer.add_scalar("Iter/Router Aux Loss", aux_loss_value_reduce, epoch_1000x)
            
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if log_writer is not None:
        log_writer.add_scalar("Epoch/Loss", loss_value_reduce, epoch)
        log_writer.add_scalar("Epoch/REC Loss", recloss_value_reduce, epoch)
        log_writer.add_scalar("Epoch/Codebook Loss", codebook_loss_value_reduce, epoch)
        if hasattr(actual_model, 'use_mod') and actual_model.use_mod and actual_model.training:
            log_writer.add_scalar("Epoch/Router Aux Loss", aux_loss_value_reduce, epoch)
            
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
        loss, rec_loss, codebook_loss, ar_loss, tk_labels, dec, unique_tokens, unique_ratio = model(x, cur_iter, step=0)
        # # Add global step counter update
        # if hasattr(model, 'module'):  # distributed training
        #     model.module.update_step_counter()
        # else:
        #     model.update_step_counter()
        # MoD aux_loss
        if hasattr(actual_model, 'use_mod') and actual_model.use_mod and actual_model.training:
            aux_loss = actual_model.last_aux_loss
            aux_loss_value = aux_loss.item()
            metric_logger.update(aux_loss=aux_loss_value)
            misc.all_reduce_mean(aux_loss_value)
            aux_loss_value_reduce = misc.all_reduce_mean(aux_loss_value)
        
        opt_ae.zero_grad() 

        lr_sched.adjust_learning_rate(opt_ae, data_iter_step / len(data_loader) + epoch, args)

        # Update parameters
        loss_scaler_ae(loss, opt_ae, parameters=list(actual_model.encoder.parameters())+
                                    list(actual_model.decoder.parameters())+
                                    list(actual_model.quant_conv.parameters())+
                                    list(actual_model.post_quant_conv.parameters())+
                                    list(actual_model.next_patch_predictor.parameters()),
                                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        torch.cuda.synchronize()
        
        lr = opt_ae.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        loss_value = loss.item()
        metric_logger.update(loss=loss_value)
        misc.all_reduce_mean(loss_value)
        loss_value_reduce = misc.all_reduce_mean(loss_value)

        rec_loss_value = rec_loss.item()
        metric_logger.update(rec_loss=rec_loss_value)
        misc.all_reduce_mean(rec_loss_value)
        rec_loss_value_reduce = misc.all_reduce_mean(rec_loss_value)

        codebook_loss_value = codebook_loss.item()
        metric_logger.update(codebook_loss=codebook_loss_value)
        misc.all_reduce_mean(codebook_loss_value)
        codebook_loss_value_reduce = misc.all_reduce_mean(codebook_loss_value)

        ar_loss_value = ar_loss.item()
        metric_logger.update(ar_loss=ar_loss_value)
        misc.all_reduce_mean(ar_loss_value)
        ar_loss_value_reduce = misc.all_reduce_mean(ar_loss_value)
         # 记录unique tokens的信息
        unique_tokens_value = float(unique_tokens)  # 将整数转换为浮点数
        metric_logger.update(unique_tokens=unique_tokens_value)
        misc.all_reduce_mean(unique_tokens_value)
        unique_tokens_value_reduce = misc.all_reduce_mean(unique_tokens_value)
        
        unique_ratio_value = unique_ratio
        metric_logger.update(unique_ratio=unique_ratio_value)
        misc.all_reduce_mean(unique_ratio_value)
        unique_ratio_value_reduce = misc.all_reduce_mean(unique_ratio_value)

        # Evaluate autoregressive generation every 1000 iterations (if enabled)
        if data_iter_step % 1000 == 0 and hasattr(args, 'eval_ar_gen') and args.eval_ar_gen:
            # Switch to evaluation mode
            model.eval()
            
            with torch.no_grad():
                # Generate images using autoregressive method
                ar_gen_img, _ = actual_model.generate(input_image=x[:4], temperature=0.0)  # No temperature, deterministic generation
                
                # Calculate reconstruction loss with original images
                ar_rec_loss = torch.mean(torch.abs(x[:4].contiguous() - ar_gen_img.contiguous()))
                metric_logger.update(ar_rec_loss=ar_rec_loss.item())
                
                # Save image comparisons
                # Prepare original images, standard reconstructions and autoregressive generated images
                save_x = (x[:4]-x[:4].min())/(x[:4].max()-x[:4].min())
                save_xrec = (dec[:4]-dec[:4].min())/(dec[:4].max()-dec[:4].min())
                save_ar = (ar_gen_img-ar_gen_img.min())/(ar_gen_img.max()-ar_gen_img.min())
                
                # Use mlflow to save images
                for b in range(save_x.shape[0]):
                    # Original images
                    mlflow.log_image(save_x[b].detach().cpu().numpy().transpose(1, 2, 0), 
                                    f"orig_{epoch}_{cur_iter}_{b}.png")
                    # Standard reconstructions
                    mlflow.log_image(save_xrec[b].detach().cpu().numpy().transpose(1, 2, 0), 
                                    f"recon_{epoch}_{cur_iter}_{b}.png")
                    # Autoregressive generations
                    mlflow.log_image(save_ar[b].detach().cpu().numpy().transpose(1, 2, 0), 
                                    f"ar_gen_{epoch}_{cur_iter}_{b}.png")
                    
                    # Combined images
                    combined = torch.cat([save_x[b], save_xrec[b], save_ar[b]], dim=2).detach().cpu().numpy()
                    mlflow.log_image(combined.transpose(1, 2, 0), 
                                    f"compare_{epoch}_{cur_iter}_{b}.png")
            
            # Switch back to training mode
            model.train()

        if log_writer is not None and cur_iter % 1000 == 0:
            epoch_1000x = int(cur_iter)
            log_writer.add_scalar("Iter/lr", lr, epoch_1000x)
            log_writer.add_scalar("Iter/Loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("Iter/Codebook Loss", codebook_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("Iter/REC Loss", rec_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("Iter/AR Loss", ar_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("Iter/Unique Tokens", unique_tokens_value_reduce, epoch_1000x)
            log_writer.add_scalar("Iter/Unique Ratio", unique_ratio_value_reduce, epoch_1000x)
            # If there's autoregressive loss, record it too
            if hasattr(metric_logger.meters, 'ar_rec_loss'):
                log_writer.add_scalar("Iter/AR REC Loss", metric_logger.meters['ar_rec_loss'].global_avg, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if log_writer is not None:
        log_writer.add_scalar("Epoch/Loss", loss_value_reduce, epoch)
        log_writer.add_scalar("Epoch/Codebook Loss", codebook_loss_value_reduce, epoch)
        log_writer.add_scalar("Epoch/REC Loss", rec_loss_value_reduce, epoch)
        log_writer.add_scalar("Epoch/AR Loss", ar_loss_value_reduce, epoch)
        log_writer.add_scalar("Epoch/Unique Tokens", unique_tokens_value_reduce, epoch)
        log_writer.add_scalar("Epoch/Unique Ratio", unique_ratio_value_reduce, epoch)
        # If there's autoregressive loss, record it too
        if 'ar_rec_loss' in metric_logger.meters:
            log_writer.add_scalar("Epoch/AR REC Loss", metric_logger.meters['ar_rec_loss'].global_avg, epoch)
    
    # Save standard reconstruction images at the end of each epoch
    save_x = (x-x.min())/(x.max()-x.min())
    save_xrec = (dec-dec.min())/(dec.max()-dec.min())
    save_img = torch.cat([save_x, save_xrec], dim=-1).detach().cpu().numpy()
    for b in range(0, min(4, save_img.shape[0])):
        mlflow.log_image(save_img[b].transpose(1, 2, 0), f"recons_{epoch}_{b}.png")
    
    # If autoregressive generation is enabled, also save autoregressive generated images at the end of each epoch
    if hasattr(args, 'eval_ar_gen') and args.eval_ar_gen:
        model.eval()
        with torch.no_grad():
            # Generate images using autoregressive method
            ar_gen_img, _ = actual_model.generate(input_image=x[:4], temperature=0.0)
            
            # Prepare images
            save_x = (x[:4]-x[:4].min())/(x[:4].max()-x[:4].min())
            save_ar = (ar_gen_img-ar_gen_img.min())/(ar_gen_img.max()-ar_gen_img.min())
            
            # Combine original images and autoregressive generated images
            save_ar_img = torch.cat([save_x, save_ar], dim=-1).detach().cpu().numpy()
            for b in range(0, min(4, save_ar_img.shape[0])):
                mlflow.log_image(save_ar_img[b].transpose(1, 2, 0), f"ar_recons_{epoch}_{b}.png")
        model.train()
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}