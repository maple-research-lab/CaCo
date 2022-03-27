import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from training.train_utils import AverageMeter, ProgressMeter, accuracy


def update_multicrop_network(model, images, args, Memory_Bank,
                       losses, top1, top5, optimizer, criterion, mem_losses,
                       moco_momentum, memory_lr, cur_adco_t):
    model.zero_grad()
    q_list, k_list = model(im_q=images[1:], im_k=images[0], run_type=1, moco_momentum=moco_momentum)
    pred_list = []
    for q_pred in q_list:
        d_norm1, d1, logits1 = Memory_Bank(q_pred)
        pred_list.append(logits1)
    # logits: Nx(1+K)
    logits_keep_list = []
    with torch.no_grad():
        for logits_tmp in pred_list:
            logits_keep_list.append(logits_tmp.clone())
    for k in range(len(pred_list)):
        pred_list[k]/=args.moco_t

    # find the positive index and label
    labels_list = []
    with torch.no_grad():
        # swap relationship, im_k supervise im_q
        for key in k_list:

            d_norm2, d2, check_logits1 = Memory_Bank(key)
            check_logits1 = check_logits1.detach()
            filter_index1 = torch.argmax(check_logits1, dim=1)
            labels1 = filter_index1
            labels_list.append(labels1)

    loss_big = 0
    loss_mini = 0
    count_big = 0
    count_mini = 0
    for i in range(len(pred_list)):

        for j in range(len(labels_list)):
            if i==j:
                continue
            if i<2:
                loss_big += criterion(pred_list[i],labels_list[j])
                count_big += 1
            else:
                loss_mini += criterion(pred_list[i],labels_list[j])
                count_mini +=1
            if i==0:
                # acc1/acc5 are (K+1)-way contrast classifier accuracy
                # measure accuracy and record loss
                acc1, acc5 = accuracy(pred_list[i], labels_list[j], topk=(1, 5))
                losses.update(loss_big.item(), images[0].size(0))
                top1.update(acc1.item(), images[0].size(0))
                top5.update(acc5.item(), images[0].size(0))
    if count_big!=0:
        loss_big = loss_big/count_big
    if count_mini!=0:
        loss_mini = loss_mini/count_mini
    loss = loss_big+loss_mini  # *args.moco_t


    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # update memory bank
    with torch.no_grad():
        # update memory bank

        g_big_sum = 0
        g_mini_sum = 0
        count_big = 0
        count_mini = 0
        for i in range(len(pred_list)):

            for j in range(len(labels_list)):
                if i == j:
                    continue
                logits1 = logits_keep_list[i]/cur_adco_t
                p_qd1 = nn.functional.softmax(logits1, dim=1)
                filter_index1 = labels_list[j]
                p_qd1[torch.arange(logits1.shape[0]), filter_index1] = 1 - p_qd1[torch.arange(logits1.shape[0]), filter_index1]
                q_pred = q_list[i]
                logits_keep1 = logits_keep_list[i]
                g1 = torch.einsum('cn,nk->ck', [q_pred.T, p_qd1]) / logits1.shape[0] - torch.mul(
                    torch.mean(torch.mul(p_qd1, logits_keep1), dim=0), d_norm1)


                g = -torch.div(g1, torch.norm(d1, dim=0))
                g /= cur_adco_t

                g = all_reduce(g) / torch.distributed.get_world_size()
                if i<2:
                    g_big_sum +=g
                    count_big +=1
                else:
                    g_mini_sum += g
                    count_mini += 1
        if count_big != 0:
            g_big_sum = g_big_sum / count_big
        if count_mini != 0:
            g_mini_sum = g_mini_sum/ count_mini
        g_sum = g_big_sum +g_mini_sum
        Memory_Bank.v.data = args.mem_momentum * Memory_Bank.v.data + g_sum  # + args.mem_wd * Memory_Bank.W.data
        Memory_Bank.W.data = Memory_Bank.W.data - memory_lr * Memory_Bank.v.data
    with torch.no_grad():
        logits1 = pred_list[0]
        filter_index1 = labels_list[0]
        logits = torch.softmax(logits1, dim=1)
        posi_prob = logits[torch.arange(logits.shape[0]), filter_index1]
        posi_prob = torch.mean(posi_prob)
        mem_losses.update(posi_prob.item(), logits.size(0))
    return pred_list

def update_sym_network(model, images, args, Memory_Bank, 
                   losses, top1, top5, optimizer, criterion, mem_losses,
                   moco_momentum,memory_lr,cur_adco_t):
    model.zero_grad()
    q_pred, k_pred, q, k = model(im_q=images[0], im_k=images[1],run_type=0,moco_momentum=moco_momentum)
    
    d_norm1, d1, logits1 = Memory_Bank(q_pred)
    d_norm2, d2, logits2 = Memory_Bank(k_pred)
    # logits: Nx(1+K)
    with torch.no_grad():
        logits_keep1 = logits1.clone()
        logits_keep2 = logits2.clone()
    logits1 /= args.moco_t #cur_adco_t#args.moco_t
    logits2 /= args.moco_t #cur_adco_t#args.moco_t
    #find the positive index and label
    with torch.no_grad():
        #swap relationship, im_k supervise im_q
        d_norm21, d21, check_logits1 = Memory_Bank(k)
        check_logits1 = check_logits1.detach()
        filter_index1 = torch.argmax(check_logits1, dim=1)
        labels1 = filter_index1

        d_norm22, d22, check_logits2 = Memory_Bank(q)
        check_logits2 = check_logits2.detach()
        filter_index2 = torch.argmax(check_logits2, dim=1)
        labels2 = filter_index2
    
    loss = (criterion(logits1, labels1)+criterion(logits2, labels2))#*args.moco_t


    # acc1/acc5 are (K+1)-way contrast classifier accuracy
    # measure accuracy and record loss
    acc1, acc5 = accuracy(logits1, labels1, topk=(1, 5))
    losses.update(loss.item(), images[0].size(0))
    top1.update(acc1.item(), images[0].size(0))
    top5.update(acc5.item(), images[0].size(0))
    acc1, acc5 = accuracy(logits2, labels2, topk=(1, 5))
    losses.update(loss.item(), images[0].size(0))
    top1.update(acc1.item(), images[0].size(0))
    top5.update(acc5.item(), images[0].size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # update memory bank
    with torch.no_grad():
        # update memory bank

        # logits: Nx(1+K)
        logits1 = logits_keep1/cur_adco_t#/args.mem_t
        # negative logits: NxK
        # logits: Nx(1+K)
        logits2 = logits_keep2/cur_adco_t#/args.mem_t
        
        p_qd1 = nn.functional.softmax(logits1, dim=1)
        p_qd1[torch.arange(logits1.shape[0]), filter_index1] = 1 - p_qd1[torch.arange(logits1.shape[0]), filter_index1]

        g1 = torch.einsum('cn,nk->ck', [q_pred.T, p_qd1]) / logits1.shape[0] - torch.mul(
            torch.mean(torch.mul(p_qd1, logits_keep1), dim=0), d_norm1)
        
        
        p_qd2 = nn.functional.softmax(logits2, dim=1)
        p_qd2[torch.arange(logits1.shape[0]), filter_index2] = 1 - p_qd2[torch.arange(logits2.shape[0]), filter_index2]



        g2 = torch.einsum('cn,nk->ck', [k_pred.T, p_qd2]) / logits2.shape[0] - torch.mul(
            torch.mean(torch.mul(p_qd2, logits_keep2), dim=0), d_norm2)
        g = -torch.div(g1, torch.norm(d1, dim=0))  - torch.div(g2,torch.norm(d2, dim=0))#/ args.mem_t  # c*k
        g /=cur_adco_t
        
        g = all_reduce(g) / torch.distributed.get_world_size()
        Memory_Bank.v.data = args.mem_momentum * Memory_Bank.v.data + g #+ args.mem_wd * Memory_Bank.W.data
        Memory_Bank.W.data = Memory_Bank.W.data - memory_lr * Memory_Bank.v.data
    with torch.no_grad():
        logits = torch.softmax(logits1, dim=1)
        posi_prob = logits[torch.arange(logits.shape[0]), filter_index1]
        posi_prob = torch.mean(posi_prob)
        mem_losses.update(posi_prob.item(), logits.size(0))
    return logits2,logits1


def update_symkey_network(model, images, args, Memory_Bank, 
                   losses, top1, top5, optimizer, criterion, mem_losses,
                   moco_momentum,memory_lr,cur_adco_t):
    model.zero_grad()
    q_pred, k_pred, q, k = model(im_q=images[0], im_k=images[1],run_type=0,moco_momentum=moco_momentum)
    
    d_norm1, d1, logits1 = Memory_Bank(q_pred)
    d_norm2, d2, logits2 = Memory_Bank(k_pred)
    # logits: Nx(1+K)
    with torch.no_grad():
        logits_keep1 = logits1.clone()
        logits_keep2 = logits2.clone()
    logits1 /= args.moco_t #cur_adco_t#args.moco_t
    logits2 /= args.moco_t #cur_adco_t#args.moco_t
    #find the positive index and label
    with torch.no_grad():
        #swap relationship, im_k supervise im_q
        d_norm21, d21, check_logits1 = Memory_Bank(k)
        check_logits1 = check_logits1.detach()
        filter_index1 = torch.argmax(check_logits1, dim=1)
        labels1 = filter_index1

        d_norm22, d22, check_logits2 = Memory_Bank(q)
        check_logits2 = check_logits2.detach()
        filter_index2 = torch.argmax(check_logits2, dim=1)
        labels2 = filter_index2
    
    loss = (criterion(logits1, labels1)+criterion(logits2, labels2))#*args.moco_t


    # acc1/acc5 are (K+1)-way contrast classifier accuracy
    # measure accuracy and record loss
    acc1, acc5 = accuracy(logits1, labels1, topk=(1, 5))
    losses.update(loss.item(), images[0].size(0))
    top1.update(acc1.item(), images[0].size(0))
    top5.update(acc5.item(), images[0].size(0))
    acc1, acc5 = accuracy(logits2, labels2, topk=(1, 5))
    losses.update(loss.item(), images[0].size(0))
    top1.update(acc1.item(), images[0].size(0))
    top5.update(acc5.item(), images[0].size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # update memory bank
    with torch.no_grad():
        # update memory bank

        # logits: Nx(1+K)
        logits1 = check_logits1/cur_adco_t#/args.mem_t
        # negative logits: NxK
        # logits: Nx(1+K)
        logits2 = check_logits2/cur_adco_t#/args.mem_t
        
        p_qd1 = nn.functional.softmax(logits1, dim=1)
        p_qd1[torch.arange(logits1.shape[0]), filter_index1] = 1 - p_qd1[torch.arange(logits1.shape[0]), filter_index1]

        g1 = torch.einsum('cn,nk->ck', [q.T, p_qd1]) / logits1.shape[0] - torch.mul(
            torch.mean(torch.mul(p_qd1, check_logits1), dim=0), d_norm21)
        
        
        p_qd2 = nn.functional.softmax(logits2, dim=1)
        p_qd2[torch.arange(logits1.shape[0]), filter_index2] = 1 - p_qd2[torch.arange(logits2.shape[0]), filter_index2]



        g2 = torch.einsum('cn,nk->ck', [k.T, p_qd2]) / logits2.shape[0] - torch.mul(
            torch.mean(torch.mul(p_qd2, check_logits2), dim=0), d_norm22)
        g = -torch.div(g1, torch.norm(d21, dim=0))  - torch.div(g2,torch.norm(d22, dim=0))#/ args.mem_t  # c*k
        g /=cur_adco_t
        
        g = all_reduce(g) / torch.distributed.get_world_size()
        Memory_Bank.v.data = args.mem_momentum * Memory_Bank.v.data + g #+ args.mem_wd * Memory_Bank.W.data
        Memory_Bank.W.data = Memory_Bank.W.data - memory_lr * Memory_Bank.v.data
    with torch.no_grad():
        logits = torch.softmax(logits1, dim=1)
        posi_prob = logits[torch.arange(logits.shape[0]), filter_index1]
        posi_prob = torch.mean(posi_prob)
        mem_losses.update(posi_prob.item(), logits.size(0))
    return logits2,logits1

def train_caco(train_loader, model, Memory_Bank, criterion,
          optimizer, epoch, args, train_log_path,moco_momentum):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    mem_losses = AverageMeter('MemLoss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, mem_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    
    if epoch<args.warmup_epochs:
        cur_memory_lr =  args.memory_lr* (epoch+1) / args.warmup_epochs 
    elif args.memory_lr != args.memory_lr_final:
        cur_memory_lr = args.memory_lr_final + 0.5 * \
                   (1. + math.cos(math.pi * (epoch-args.warmup_epochs) / (args.epochs-args.warmup_epochs))) \
                   * (args.memory_lr- args.memory_lr_final)
    else:
        cur_memory_lr = args.memory_lr
    print("current memory lr %f"%cur_memory_lr)
    cur_adco_t =args.mem_t
    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
        batch_size = images[0].size(0)
        if args.multi_crop:
            update_multicrop_network(model, images, args, Memory_Bank, losses, top1, top5,
                               optimizer, criterion, mem_losses, moco_momentum, cur_memory_lr, cur_adco_t)
        else:
            update_sym_network(model, images, args, Memory_Bank, losses, top1, top5,
            optimizer, criterion, mem_losses,moco_momentum,cur_memory_lr,cur_adco_t)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.rank==0:
            progress.display(i)
            if args.rank == 0:
                progress.write(train_log_path, i)
    return top1.avg



@torch.no_grad()
def all_reduce(tensor):
    """
    Performs all_reduce(mean) operation on the provided tensors.
    *** Warning ***: torch.distributed.all_reduce has no gradient.
    """
    torch.distributed.all_reduce(tensor, async_op=False)

    return tensor


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output