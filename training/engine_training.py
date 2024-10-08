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

    opt_ae, opt_disc = optimizer
    loss_scaler_ae, loss_scaler_disc = loss_scaler
    #optimizer.zero_grad()
    token_freq = torch.zeros(args.n_vision_words).to(device)

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))
    for data_iter_step, [image_ids, images, clip_images, label_cls] in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
    #for data_iter_step, images in enumerate(data_loader):

        cur_iter = len(data_loader) * epoch + data_iter_step

        ####Tokenizer with VQ-GAN
        b = images.shape[0]
        x = images.to(device)
        label_cls = label_cls.to(device)

        loss, rec_loss, qloss, p_loss, g_loss, tk_labels, xrec = model(x, cur_iter, step=0)
        
        tk_index_one_hot = torch.nn.functional.one_hot(tk_labels.view(-1), num_classes=args.n_vision_words)
        tk_index_num = torch.sum(tk_index_one_hot, dim=0)
        token_freq += tk_index_num
        
        opt_ae.zero_grad()
        lr_sched.adjust_learning_rate(opt_ae, data_iter_step / len(data_loader) + epoch, args)

        if args.use_cblinear == 1:
            loss_scaler_ae(loss, opt_ae, parameters=list(model.module.encoder.parameters())+
                                    list(model.module.decoder.parameters())+
                                    list(model.module.quant_conv.parameters())+
                                    list(model.module.tok_embeddings.parameters())+
                                    list(model.module.codebook_projection.parameters()) + 
                                    list(model.module.post_quant_conv.parameters()), update_grad=(data_iter_step + 1) % accum_iter == 0)
        else:
            loss_scaler_ae(loss, opt_ae, parameters=list(model.encoder.parameters())+
                                        list(model.decoder.parameters())+
                                        list(model.quant_conv.parameters())+
                                        list(model.tok_embeddings.parameters())+
                                        list(model.post_quant_conv.parameters()), 
                                        update_grad=(data_iter_step + 1) % accum_iter == 0)
        if cur_iter > args.disc_start and args.rate_d != 0:
            d_loss, _, _, _, _, _, _ = model(x, cur_iter, step=1)
            opt_disc.zero_grad()
            lr_sched.adjust_learning_rate(opt_disc, data_iter_step / len(data_loader) + epoch, args)
            loss_scaler_disc(d_loss, opt_disc, parameters=model.discriminator.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)

        torch.cuda.synchronize()
        
        lr = opt_ae.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        loss_value = loss.item()
        metric_logger.update(loss=loss_value)
        misc.all_reduce_mean(loss_value)
        loss_value_reduce = misc.all_reduce_mean(loss_value)

        #discloss_value = discloss.item()
        recloss_value = rec_loss.item()
        metric_logger.update(recloss=recloss_value)
        misc.all_reduce_mean(recloss_value)
        recloss_value_reduce = misc.all_reduce_mean(recloss_value)

        gloss_value = g_loss.item()
        metric_logger.update(gloss=gloss_value)
        misc.all_reduce_mean(gloss_value)
        gloss_value_reduce = misc.all_reduce_mean(gloss_value)

        if cur_iter > args.disc_start and args.rate_d != 0:
            dloss_value = d_loss.item()
            metric_logger.update(dloss=dloss_value)
            misc.all_reduce_mean(dloss_value)
            dloss_value_reduce = misc.all_reduce_mean(dloss_value)

        p_loss_value = p_loss.item()
        metric_logger.update(p_loss=p_loss_value)
        misc.all_reduce_mean(p_loss_value)
        p_loss_value_reduce = misc.all_reduce_mean(p_loss_value)

        qloss_value = qloss.item()
        metric_logger.update(qloss=qloss_value)
        misc.all_reduce_mean(qloss_value)
        qloss_value_reduce = misc.all_reduce_mean(qloss_value)


        """We use epoch_1000x as the x-axis in tensorboard.
        This calibrates different curves when batch size changes.
        """
        if log_writer is not None and cur_iter % 1000 == 0:
            epoch_1000x = int(cur_iter)
            log_writer.add_scalar("Iter/lr", lr, epoch_1000x)
            log_writer.add_scalar("Iter/Loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("Iter/REC Loss", recloss_value_reduce, epoch_1000x)
            log_writer.add_scalar("Iter/Q Loss", qloss_value_reduce, epoch_1000x)
            log_writer.add_scalar("Iter/VGG Loss", p_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("Iter/GAN Loss", gloss_value_reduce, epoch_1000x)
            if cur_iter > args.disc_start and args.rate_d != 0:
                log_writer.add_scalar("Iter/Discriminator Loss", dloss_value_reduce, epoch_1000x)
    
    efficient_token = np.sum(np.array(token_freq.cpu().data) != 0)
    #metric_logger.update(efficient_token=efficient_token.float())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print("Efficient Tokens:", efficient_token)
    if log_writer is not None:
        log_writer.add_scalar("Epoch/Loss", loss_value_reduce, epoch)
        log_writer.add_scalar("Epoch/REC Loss", recloss_value_reduce, epoch)
        log_writer.add_scalar("Epoch/Q Loss", qloss_value_reduce, epoch)
        log_writer.add_scalar("Epoch/VGG Loss", p_loss_value_reduce, epoch)

        log_writer.add_scalar("Epoch/GAN Loss", gloss_value_reduce, epoch)
        if cur_iter > args.disc_start and args.rate_d != 0:
            log_writer.add_scalar("Epoch/Discriminator Loss", dloss_value_reduce, epoch)
            
        log_writer.add_scalar("Efficient Token", efficient_token, epoch)
        save_x = (x-x.min())/(x.max()-x.min())#self.to_rgb(x)
        save_xrec = (xrec-xrec.min())/(xrec.max()-xrec.min())
        save_img = torch.cat([save_x, save_xrec], dim=-1).detach().cpu().numpy()
        for b in range(0, save_img.shape[0]):
            mlflow.log_image(save_img[b].transpose(1, 2, 0), "recons_%s_%s.png"%(epoch, b))
    


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_stage2(
    llama: torch.nn.Module,
    vqgan: torch.nn.Module,
    classifier: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    metric_logger = misc.MetricLogger(delimiter="  ")
    # metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("acc1", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("acc5", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    header = "Epoch: [{}]".format(epoch)
    print_freq = 10
    # classifier.train()
    accum_iter = args.accum_iter
    criterion = torch.nn.CrossEntropyLoss()
    classifier_para = optimizer
    loss_scaler = loss_scaler
    # for data_iter_step, (features, labels, image_ids) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    #     b = features.shape[0]
    #     cur_iter = len(data_loader) * epoch + data_iter_step
    #     features, labels = features.to(device), labels.to(device)
    #     outputs = classifier(features)
    #     loss = criterion(outputs, labels)
    #     acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))    
    for data_iter_step, (image_ids, images, clip_images, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        b = images.shape[0]
        cur_iter = len(data_loader) * epoch + data_iter_step
        images, labels = images.to(device), labels.to(device)
        visual_embeddings, visual_codebook_indices = vqgan(images, data_iter_step, step=0, is_val=True)
        ############# if using codebook_projector #############
        # tok_embeddings_weights = llama.embed_tokens.weight  # Get the weight tensor from the Embedding
        # projected_tok_embeddings = projector(tok_embeddings_weights)
        
        # visual_embeddings_flattened = visual_embeddings.permute(0, 2, 3, 1).reshape(-1, 768)
        
        # # Compute distances
        # d = torch.sum(visual_embeddings_flattened ** 2, dim=1, keepdim=True) + \
        #     torch.sum(projected_tok_embeddings**2, dim=1) - 2 * \
        #     torch.einsum('bd,dn->bn', visual_embeddings_flattened, rearrange(projected_tok_embeddings, 'n d -> d n'))
        
        # min_encoding_indices = torch.argmin(d, dim=1)

        ######## Apply average pooling to reduce dimensions from (bs, 768, 16, 16) to (bs, 768)
        # visual_embeddings = torch.nn.functional.adaptive_avg_pool2d(visual_embeddings, (1, 1)).squeeze(-1).squeeze(-1)
        
        # Get closest projected token embeddings
        # closest_tok_embeddings = projected_tok_embeddings[min_encoding_indices]
        
        # Compute commitment loss
        # projector_loss = torch.mean((closest_tok_embeddings.detach() - visual_embeddings_flattened)**2) + \
        #                  0.33 * torch.mean((closest_tok_embeddings - visual_embeddings_flattened.detach())**2)
        
        
        # Use the indices to get LLaMA embeddings
        # llama_inputs = min_encoding_indices.view(b, -1)  # Reshape to (batch_size, sequence_length)
        llama_outputs = llama(visual_codebook_indices)
        # print("llama_outputs.last_hidden_state.shape", llama_outputs.last_hidden_state.shape)
        # Compute classification loss
        prediction = classifier(llama_outputs.last_hidden_state[:, -1, :])
        loss = criterion(prediction, labels)
        optimizer.zero_grad()
        loss_scaler(loss, optimizer,parameters=classifier.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        # lr = optimizer.param_groups[0]["lr"]
        # Update metrics
        acc1, acc5 = accuracy(prediction, labels, topk=(1, 5))

        metric_logger.update(loss=loss.item())
        # metric_logger.update(projector_loss=projector_loss.item())
        # metric_logger.update(classification_loss=classification_loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=b)
        metric_logger.meters['acc5'].update(acc5.item(), n=b)
        """We use epoch_1000x as the x-axis in tensorboard.
        This calibrates different curves when batch size changes.
        """
        if log_writer is not None and cur_iter % 1000 == 0:
            epoch_1000x = int(cur_iter)
            # log_writer.add_scalar("Iter/lr", lr, epoch_1000x)
            log_writer.add_scalar("Iter/Loss", loss, epoch_1000x)
            log_writer.add_scalar("Iter/Acc1", acc1.item(), epoch_1000x)
            log_writer.add_scalar("Iter/Acc5", acc5.item(), epoch_1000x)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if log_writer is not None:
        log_writer.add_scalar("Epoch/Loss", loss, epoch)
        log_writer.add_scalar("Epoch/Acc1", metric_logger.meters['acc1'].global_avg, epoch)
        log_writer.add_scalar("Epoch/Acc5", metric_logger.meters['acc5'].global_avg, epoch)


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(vqgan, llama, classifier, data_loader, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 10
    
    # switch to evaluation mode
    vqgan.eval()
    llama.eval()
    # projector.eval()
    classifier.eval()

    for data_iter_step, (image_ids, images, clip_images, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images, labels = images.to(device), labels.to(device)
        batch_size = images.shape[0]
        
        # compute output
        with torch.cuda.amp.autocast():
            visual_embeddings, visual_codebook_indices = vqgan(images, data_iter_step, step=0, is_val=True)
            llama_outputs = llama(visual_codebook_indices)
            # # Project token embeddings
            # tok_embeddings_weights = vqgan.module.tok_embeddings.weight.detach()
            # projected_tok_embeddings = projector(tok_embeddings_weights)
            
            # visual_embeddings_flattened = visual_embeddings.permute(0, 2, 3, 1).reshape(-1, 768)
            
            # # Compute distances
            # d = torch.sum(visual_embeddings_flattened ** 2, dim=1, keepdim=True) + \
            #     torch.sum(projected_tok_embeddings**2, dim=1) - 2 * \
            #     torch.einsum('bd,dn->bn', visual_embeddings_flattened, rearrange(projected_tok_embeddings, 'n d -> d n'))
            
            # min_encoding_indices = torch.argmin(d, dim=1)
            
            # # Use the indices to get LLaMA embeddings
            # llama_inputs = min_encoding_indices.view(batch_size, -1)  # Reshape to (batch_size, sequence_length)
            # llama_outputs = llama(llama_inputs)
            
            # Compute classification prediction
            prediction = classifier(llama_outputs.last_hidden_state[:, -1, :])
            
            loss = criterion(prediction, labels)

        acc1, acc5 = accuracy(prediction, labels, topk=(1, 5))

        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('Eval loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
def save_llama_outputs(
    vqgan: torch.nn.Module,
    llama: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    output_dir: str,
):
    llama.eval()
    vqgan.eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (image_ids, images, clip_images, labels) in enumerate(tqdm(data_loader)):
            images = images.to(device)
            labels = labels.to(device)
            
            # VQGAN forward pass
            visual_embeddings, vqgan_indices = vqgan(images, 0, step=0, is_val=True)
            
            # LLaMA forward pass
            llama_outputs = llama(vqgan_indices)
            features = llama_outputs.last_hidden_state[:, -1, :].cpu().numpy() # last token as features

            ######################## code for visualization ########################
            # bs, dim, h, w = visual_embeddings.shape
            # visual_embeddings_flat = visual_embeddings.permute(0, 2, 3, 1).reshape(bs * h * w, dim).cpu().numpy()

            # tok_embeddings = model.module.tok_embeddings.weight.detach().cpu().numpy()
            # pca_tok = PCA(n_components=2)
            # tok_embeddings_pca = pca_tok.fit_transform(tok_embeddings)

            # pca_visual = PCA(n_components=2)
            # visual_embeddings_pca = pca_visual.fit_transform(visual_embeddings_flat)
            
            # pca_llama = PCA(n_components=2)
            # llama_embeddings_pca = pca_llama.fit_transform(llama_embeddings)

            
            # # Plotting the embeddings
            # plt.figure(figsize=(12, 10))
            # plt.scatter(tok_embeddings_pca[:, 0], tok_embeddings_pca[:, 1], label='Codebook Embeddings', alpha=0.6, c='blue')
            # plt.scatter(visual_embeddings_pca[:, 0], visual_embeddings_pca[:, 1], label='Visual Embeddings', alpha=0.6, c='red')
            # plt.scatter(llama_embeddings_pca[:, 0], llama_embeddings_pca[:, 1], label='LLaMA Embeddings', alpha=0.6, c='green')
            # plt.title('PCA Visualization of Embedding Spaces')
            # plt.xlabel('Principal Component 1')
            # plt.ylabel('Principal Component 2')
            # plt.legend()
            # plt.savefig('/home/zhiqiang.shen/projects/visual_tokenizer/V2L-Tokenizer/embedding_visualization.png')

            # exit()
            
            
            
            
            ############### Save features and labels
            for i in range(features.shape[0]):
                feature_filename = f"{output_dir}/feature_{batch_idx}_{i}.npy"
                label_filename = f"{output_dir}/label_{batch_idx}_{i}.npy"
                
                np.save(feature_filename, features[i])
                np.save(label_filename, labels[i].cpu().numpy())
                
                # Save image_id for additional verification
                image_id_filename = f"{output_dir}/image_id_{batch_idx}_{i}.txt"
                with open(image_id_filename, 'w') as f:
                    f.write(image_ids[i])

    print(f"Saved {len(data_loader) * data_loader.batch_size} samples to {output_dir}")