import math
import sys
from typing import Iterable
from tqdm import tqdm  
import torch
import util.lr_sched as lr_sched
import util.misc as misc
import copy
from timm.utils import accuracy
import numpy as np
import mlflow
from einops import rearrange
import matplotlib.pyplot as plt
import os
from torch.nn import functional as F
from training import utils
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
    ##
    #metric_logger.add_meter("acc", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    ##
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    opt_ae = optimizer
    loss_scaler_ae = loss_scaler
    #optimizer.zero_grad()
    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))
    for data_iter_step, [image_ids, images, clip_images, label_cls] in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        cur_iter = len(data_loader) * epoch + data_iter_step

        ####Tokenizer with VQ-GAN
        b = images.shape[0]
        x = images.to(device)
        label_cls = label_cls.to(device)

        loss, rec_loss, codebook_loss, p_loss, next_token_loss, tk_labels, dec = model(x, cur_iter, step=0)
        opt_ae.zero_grad() 

        lr_sched.adjust_learning_rate(opt_ae, data_iter_step / len(data_loader) + epoch, args)

        loss_scaler_ae(loss, opt_ae, parameters=list(model.encoder.parameters())+
                                    list(model.decoder.parameters())+
                                    list(model.quant_conv.parameters())+
                                    list(model.post_quant_conv.parameters()), 
                                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        # if cur_iter > args.disc_start and args.rate_d != 0:
        #     d_loss, _, _,_, _, _, _ = model(x, cur_iter, step=1)
        #     opt_disc.zero_grad()
        #     lr_sched.adjust_learning_rate(opt_disc, data_iter_step / len(data_loader) + epoch, args)
        #     loss_scaler_disc(d_loss, opt_disc, parameters=model.discriminator.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)

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

        next_token_loss_value = next_token_loss.item()
        metric_logger.update(next_token_loss=next_token_loss_value)
        misc.all_reduce_mean(next_token_loss_value)
        next_token_loss_value_reduce = misc.all_reduce_mean(next_token_loss_value)

        # if cur_iter > args.disc_start and args.rate_d != 0:
        #     dloss_value = d_loss.item()
        #     metric_logger.update(dloss=dloss_value)
        #     misc.all_reduce_mean(dloss_value)
        #     dloss_value_reduce = misc.all_reduce_mean(dloss_value)

        p_loss_value = p_loss.item()
        metric_logger.update(p_loss=p_loss_value)
        misc.all_reduce_mean(p_loss_value)
        p_loss_value_reduce = misc.all_reduce_mean(p_loss_value)
        codebook_loss_value = codebook_loss.item()
        metric_logger.update(codebook_loss=codebook_loss_value)
        misc.all_reduce_mean(codebook_loss_value)
        codebook_loss_value_reduce = misc.all_reduce_mean(codebook_loss_value)

        """We use epoch_1000x as the x-axis in tensorboard.
        This calibrates different curves when batch size changes.
        """
        if log_writer is not None and cur_iter % 1000 == 0:
            epoch_1000x = int(cur_iter)
            log_writer.add_scalar("Iter/lr", lr, epoch_1000x)
            log_writer.add_scalar("Iter/Loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("Iter/REC Loss", recloss_value_reduce, epoch_1000x)
            log_writer.add_scalar("Iter/Codebook Loss", codebook_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("Iter/VGG Loss", p_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("Iter/Next Token Loss", next_token_loss_value_reduce, epoch_1000x)
            # if cur_iter > args.disc_start and args.rate_d != 0:
            #     log_writer.add_scalar("Iter/Discriminator Loss", dloss_value_reduce, epoch_1000x)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if log_writer is not None:
        log_writer.add_scalar("Epoch/Loss", loss_value_reduce, epoch)
        log_writer.add_scalar("Epoch/REC Loss", recloss_value_reduce, epoch)
        log_writer.add_scalar("Epoch/VGG Loss", p_loss_value_reduce, epoch)
        log_writer.add_scalar("Epoch/Codebook Loss", codebook_loss_value_reduce, epoch)
        log_writer.add_scalar("Epoch/Next Token Loss", next_token_loss_value_reduce, epoch)
        # if cur_iter > args.disc_start and args.rate_d != 0:
        #     log_writer.add_scalar("Epoch/Discriminator Loss", dloss_value_reduce, epoch)
            
        save_x = (x-x.min())/(x.max()-x.min())#self.to_rgb(x)
        save_xrec = (dec-dec.min())/(dec.max()-dec.min())
        save_img = torch.cat([save_x, save_xrec], dim=-1).detach().cpu().numpy()
        for b in range(0, save_img.shape[0]):
            mlflow.log_image(save_img[b].transpose(1, 2, 0), "recons_%s_%s.png"%(epoch, b))
    


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}