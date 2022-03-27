import torch.nn.functional as F
import torch
import torch.nn as nn
# code copied from https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb#scrollTo=RI1Y8bSImD7N
# test using a knn monitor
def knn_monitor(net, memory_data_loader, test_data_loader,
                global_k=200,pool_ops=True,temperature=0.2,
                vit_backbone=False):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    feature_labels=[]
    avgpool = nn.AdaptiveAvgPool2d((1, 1))
    with torch.no_grad():
        # generate feature bank
        for k,(data, target) in enumerate(memory_data_loader):

            target = target.cuda(non_blocking=True)
            if vit_backbone:
                feature = net(data.cuda(non_blocking=True),feature_only=True)
            else:
                feature = net(data.cuda(non_blocking=True))
                if pool_ops:
                    feature = avgpool(feature)
                feature = torch.flatten(feature, 1)
            feature = F.normalize(feature, dim=1)
            feature = concat_all_gather(feature)
            feature_bank.append(feature)
            target = concat_all_gather(target)
            feature_labels.append(target)
            print("KNN feature accumulation %d/%d"%(k,len(memory_data_loader)))
        # [D, N]
        torch.cuda.empty_cache()
        print("gpu consuming before combining:", torch.cuda.memory_allocated() / 1024 / 1024)
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        print("feature bank size: ",feature_bank.size())
        # [N]
        #feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=device)
        feature_labels = torch.cat(feature_labels, dim=0).contiguous()
        print("feature label size:",feature_labels.size())
        # loop test data to predict the label by weighted knn search
        test_bar = enumerate(test_data_loader)
        for  k,(data, target) in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            if vit_backbone:
                feature = net(data, feature_only=True)
            else:
                feature = net(data)
                if pool_ops:
                    feature = avgpool(feature)
                feature = torch.flatten(feature, 1)
            feature = F.normalize(feature, dim=1)
            feature = concat_all_gather(feature)
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, global_k,temperature)
            #concat data in other gpus
            #pred_labels = concat_all_gather(pred_labels)
            target = concat_all_gather(target)
            total_num += target.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            print("current eval feature size: ",feature.size())
            print({'#KNN monitor Accuracy': total_top1 / total_num * 100})
    del feature_bank
    del feature_labels
    return total_top1 / total_num * 100

def knn_monitor_center3(net, memory_data_loader, test_data_loader, global_k=200,temperature=0.2):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    feature_labels=[]
    avgpool = nn.AdaptiveAvgPool2d((1, 1))
    with torch.no_grad():
        # generate feature bank
        for k,(data, target) in enumerate(memory_data_loader):
            feature = net(data.cuda(non_blocking=True))
            target = target.cuda(non_blocking=True)
            feature = feature[:,:,2:5,2:5]
            feature = torch.mean(feature,dim=3)
            feature = torch.mean(feature, dim=2)
            #feature = torch.flatten(feature, 1)
            feature = F.normalize(feature, dim=1)
            feature = concat_all_gather(feature)
            feature_bank.append(feature)
            target = concat_all_gather(target)
            feature_labels.append(target)
            print("KNN feature accumulation %d/%d"%(k,len(memory_data_loader)))
        # [D, N]
        torch.cuda.empty_cache()
        print("gpu consuming before combining:", torch.cuda.memory_allocated() / 1024 / 1024)
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        print("feature bank size: ",feature_bank.size())
        # [N]
        #feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=device)
        feature_labels = torch.cat(feature_labels, dim=0).contiguous()
        print("feature label size:",feature_labels.size())
        # loop test data to predict the label by weighted knn search
        test_bar = enumerate(test_data_loader)
        for  k,(data, target) in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = feature[:, :, 2:5, 2:5]
            feature = torch.mean(feature, dim=3)
            feature = torch.mean(feature, dim=2)
            #feature = torch.flatten(feature, 1)
            feature = F.normalize(feature, dim=1)
            feature = concat_all_gather(feature)
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, global_k,temperature)
            #concat data in other gpus
            #pred_labels = concat_all_gather(pred_labels)
            target = concat_all_gather(target)
            total_num += target.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            print("current eval feature size: ",feature.size())
            print({'#KNN monitor Accuracy': total_top1 / total_num * 100})
    del feature_bank
    del feature_labels
    return total_top1 / total_num * 100


def knn_monitor_fast(net, memory_data_loader, test_data_loader, global_k=200,temperature=0.2):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    feature_labels=[]
    avgpool = nn.AdaptiveAvgPool2d((1, 1))
    with torch.no_grad():
        # generate feature bank
        for k,(data, target) in enumerate(memory_data_loader):
            feature = net(data.cuda(non_blocking=True))
            target = target.cuda(non_blocking=True)
            feature = avgpool(feature)
            feature = torch.flatten(feature, 1)
            feature = F.normalize(feature, dim=1)
            #feature = concat_all_gather(feature)
            feature_bank.append(feature)
            #target = concat_all_gather(target)
            feature_labels.append(target)
            print("KNN feature accumulation %d/%d"%(k,len(memory_data_loader)))
        # [D, N]
        torch.cuda.empty_cache()
        print("gpu consuming before combining:", torch.cuda.memory_allocated() / 1024 / 1024)
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        print("feature bank size: ",feature_bank.size())
        # [N]
        #feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=device)
        feature_labels = torch.cat(feature_labels, dim=0).contiguous()
        print("feature label size:",feature_labels.size())
        # loop test data to predict the label by weighted knn search
        test_bar = enumerate(test_data_loader)
        for  k,(data, target) in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = avgpool(feature)
            feature = torch.flatten(feature, 1)
            feature = F.normalize(feature, dim=1)
            #feature = concat_all_gather(feature)
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, global_k,temperature)
            #concat data in other gpus
            pred_labels = pred_labels.contiguous()
            pred_labels = concat_all_gather(pred_labels)
            target = concat_all_gather(target)
            total_num += target.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            print("current eval feature size: ",feature.size())
            print({'#KNN monitor Accuracy': total_top1 / total_num * 100})
    del feature_bank
    del feature_labels
    return total_top1 / total_num * 100
# utils
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

# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k,knn_t=None):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.einsum("nc,mc->nm",[feature,feature_bank])#torch.mm(feature, feature_bank)
    if knn_t is not None:
        sim_matrix = (sim_matrix / knn_t).exp()
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)

    # counts for each class
    one_hot_label = torch.zeros((feature.size(0) * knn_k, classes), device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

def knn_monitor_horovod(net, memory_data_loader, test_data_loader,
                global_k=200,pool_ops=True,temperature=0.2,
                vit_backbone=False):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    feature_labels=[]
    avgpool = nn.AdaptiveAvgPool2d((1, 1))
    with torch.no_grad():
        # generate feature bank
        for k,(data, target) in enumerate(memory_data_loader):

            target = target.cuda(non_blocking=True)
            if vit_backbone:
                feature = net(data.cuda(non_blocking=True),feature_only=True)
            else:
                feature = net(data.cuda(non_blocking=True))
                if pool_ops:
                    feature = avgpool(feature)
                feature = torch.flatten(feature, 1)
            feature = F.normalize(feature, dim=1)
            feature = concat_all_gather2(feature)
            feature_bank.append(feature)
            target = concat_all_gather2(target)
            feature_labels.append(target)
            print("KNN feature accumulation %d/%d"%(k,len(memory_data_loader)))
        # [D, N]
        torch.cuda.empty_cache()
        print("gpu consuming before combining:", torch.cuda.memory_allocated() / 1024 / 1024)
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        print("feature bank size: ",feature_bank.size())
        # [N]
        #feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=device)
        feature_labels = torch.cat(feature_labels, dim=0).contiguous()
        print("feature label size:",feature_labels.size())
        # loop test data to predict the label by weighted knn search
        test_bar = enumerate(test_data_loader)
        for  k,(data, target) in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            if vit_backbone:
                feature = net(data, feature_only=True)
            else:
                feature = net(data)
                if pool_ops:
                    feature = avgpool(feature)
                feature = torch.flatten(feature, 1)
            feature = F.normalize(feature, dim=1)
            feature = concat_all_gather2(feature)
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, global_k,temperature)
            #concat data in other gpus
            #pred_labels = concat_all_gather(pred_labels)
            target = concat_all_gather2(target)
            total_num += target.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            print("current eval feature size: ",feature.size())
            print({'#KNN monitor Accuracy': total_top1 / total_num * 100})
    del feature_bank
    del feature_labels
    return total_top1 / total_num * 100
@torch.no_grad()
def concat_all_gather2(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    from horovod.torch.mpi_ops import allgather
    return allgather(tensor)