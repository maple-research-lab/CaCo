# CaCo

<a href="https://github.com/marktext/marktext/releases/latest">
   <img src="https://img.shields.io/badge/CaCo-v1.0.0-green">
   <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Mac%20-green">
   <img src="https://img.shields.io/badge/Language-python3-green">
   <img src="https://img.shields.io/badge/dependencies-tested-green">
   <img src="https://img.shields.io/badge/licence-MIT-green">
</a>   

CaCo is a contrastive-learning based self-supervised learning methods, which is submitted to IEEE-T-PAMI.

Copyright (C) 2020 Xiao Wang, Yuhang Huang, Dan Zeng, Guo-Jun Qi

License: MIT for academic use.

Contact: Xiao Wang (wang3702@purdue.edu), Guo-Jun Qi (guojunq@gmail.com)

<p align="center">
  <img src="https://github.com/maple-research-lab/CaCo/blob/main/caco_diagram.png" width="300">
</p>

## Introduction

As a representative self-supervised method, contrastive learning has achieved great successes in unsupervised training of representations. It trains an encoder by distinguishing positive samples from negative ones given query anchors. These positive and negative samples play critical roles in defining the objective to learn the discriminative encoder, avoiding it from learning trivial features. While existing methods heuristically choose these samples, we present a principled method where both positive and negative samples are directly learnable end-to-end with the encoder. We show that the positive and negative samples can be cooperatively and adversarially learned by minimizing and maximizing the contrastive loss, respectively. This yields cooperative positives and adversarial negatives with respect to the encoder, which are updated to continuously track the learned representation of the query anchors over mini-batches. The proposed method achieves 72.0% and 75.3% in top-1 accuracy respectively over 200 and 800 epochs of pre-training ResNet-50 backbone on ImageNet1K without tricks such as multi-crop or stronger augmentations. With Multi-Crop, it can be further boosted into 75.7%.


## Installation  
CUDA version should be 10.1 or higher. 
### 1. [`Install git`](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 
### 2. Clone the repository in your computer 
```
git clone git@github.com:maple-research-lab/CaCo.git && cd CaCo
```

### 3. Build dependencies.   
You have two options to install dependency on your computer:
#### 3.1 Install with pip and python(Ver 3.6.9).
##### 3.1.1[`install pip`](https://pip.pypa.io/en/stable/installing/).
##### 3.1.2  Install dependency in command line.
```
pip install -r requirements.txt --user
```
If you encounter any errors, you can install each library one by one:
```
pip install torch>=1.7.1
pip install torchvision>=0.8.2
pip install numpy>=1.19.5
pip install Pillow>=5.1.0
pip install tensorboard>=1.14.0
pip install tensorboardX>=1.7
```

#### 3.2 Install with anaconda
##### 3.2.1 [`install conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html). 
##### 3.2.2 Install dependency in command line
```
conda create -n CaCo python=3.7.1
conda activate CaCo
pip install -r requirements.txt 
```
Each time when you want to run my code, simply activate the environment by
```
conda activate CaCo
conda deactivate(If you want to exit) 
```
#### 4 Prepare the ImageNet dataset
##### 4.1 Download the [ImageNet2012 Dataset](http://image-net.org/challenges/LSVRC/2012/) under "./datasets/imagenet2012".
##### 4.2 Go to path "./datasets/imagenet2012/val"
##### 4.3 move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Usage

### 1. Single-Crop Unsupervised Pre-Training
#### 1.1 Training with batch size of 1024 (Single Machine)
For batch-size of 1024, we can run on a single machine of 8\*V100 32gb GPU with the following command:
```
python3 main.py --type=0 --lr=0.3 --lr_final=0.003 --memory_lr=3.0 --memory_lr_final=3.0 --cluster=65536 --moco_t=0.08 --mem_t=0.08 --data=datasets/imagenet --dist_url=tcp://localhost:10001 --batch_size=1024 --wd=1.5e-6 --mem_wd=0 --moco_dim=256 --moco_m=0.99 --moco_m_decay=1 --mlp_dim=2048 --epochs=200 --warmup_epochs=10 --nodes_num=1 --workers=32 --world_size 1 --rank=0 --mem_momentum=0.9 --ad_init=1 --knn_batch_size=1024 --multi_crop=0 --knn_freq=10
```
This should be able to reproduce our 71.3% performance with batch size 1024.

#### 1.2 Training with batch size of 4096 (4 Machines)
This can only run with multiple machines. Limited by our computing resources, we run experiments with 2048 on 4 8\*V100 GPU matchines
On the first node machine, run the following command:
```
python3 main.py --type=0 --lr=0.3 --lr_final=0.003 --memory_lr=3.0 --memory_lr_final=3.0 --cluster=65536 --moco_t=0.08 --mem_t=0.08 --data=datasets/imagenet --dist_url=tcp://localhost:10001 --batch_size=4096 --wd=1.5e-6 --mem_wd=0 --moco_dim=256 --moco_m=0.99 --moco_m_decay=1 --mlp_dim=2048 --epochs=200 --warmup_epochs=10 --nodes_num=1 --workers=128 --world_size 4 --rank=0 --mem_momentum=0.9 --ad_init=1 --knn_batch_size=1024 --multi_crop=0 --knn_freq=20
```
Then iteratively run on other nodes with the following command:
```
python3 main.py --type=0 --lr=0.3 --lr_final=0.003 --memory_lr=3.0 --memory_lr_final=3.0 --cluster=65536 --moco_t=0.08 --mem_t=0.08 --data=datasets/imagenet --dist_url=tcp://[master_id]:10001 --batch_size=4096 --wd=1.5e-6 --mem_wd=0 --moco_dim=256 --moco_m=0.99 --moco_m_decay=1 --mlp_dim=2048 --epochs=200 --warmup_epochs=10 --nodes_num=1 --workers=128 --world_size 4 --rank=[rank_id] --mem_momentum=0.9 --ad_init=1 --knn_batch_size=1024 --multi_crop=0 --knn_freq=20
```
Here we should change [master_ip] to the IP of the 1st node, also we should adjust rank with 1, 2, and 3 for 3 different nodes.


### 2. Multi-Crop Unsupervised Pre-Training (4 Machines)
This can only be run with multiple machines. Limited by our computing resources, we run experiments with 2048 on 4 8\*V100 GPU matchines
On the first node machine, run the following command:
```
python3 main.py --type=0 --lr=0.3 --lr_final=0.003 --memory_lr=3.0 --memory_lr_final=3.0 --cluster=65536 --moco_t=0.08 --mem_t=0.08 --data=datasets/imagenet --dist_url=tcp://localhost:10001 --batch_size=2048 --wd=1.5e-6 --mem_wd=0 --moco_dim=256 --moco_m=0.99 --moco_m_decay=1 --mlp_dim=2048 --epochs=800 --warmup_epochs=10 --nodes_num=4 --workers=128 --world_size 4 --rank=0 --mem_momentum=0.9 --ad_init=1 --knn_batch_size=2048 --multi_crop=1 --knn_freq=50
```
Then iteratively run on other nodes with the following command:
```
python3 main.py --type=0 --lr=0.3 --lr_final=0.003 --memory_lr=3.0 --memory_lr_final=3.0 --cluster=65536 --moco_t=0.08 --mem_t=0.08 --data=datasets/imagenet --dist_url=tcp://[master_id]:10001 --batch_size=2048 --wd=1.5e-6 --mem_wd=0 --moco_dim=256 --moco_m=0.99 --moco_m_decay=1 --mlp_dim=2048 --epochs=800 --warmup_epochs=10 --nodes_num=4 --workers=128 --world_size 4 --rank=[rank_id] --mem_momentum=0.9 --ad_init=1 --knn_batch_size=2048 --multi_crop=1 --knn_freq=50
```
Here we should change [master_ip] to the IP of the 1st node, also we should adjust rank with 1, 2, and 3 for 3 different nodes.<br>
We believe further increase the batch size to 4096 can increase the performance.

### Linear Classification
With a pre-trained model, we can easily evaluate its performance on ImageNet with:
```
python linear.py  -a resnet50 --lr 0.025 --batch-size 4096 \
  --pretrained [your checkpoint path] \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed \
  --world-size 1 --rank 0 --data [imagenet path]
```


### Linear Performance:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">pre-train<br/>network</th>
<th valign="bottom">pre-train<br/>epochs</th>
<th valign="bottom">Crop</th>
<th valign="bottom">Batch<br/>Size</th>
<th valign="bottom">CaCo<br/>top-1 acc.</th>
<th valign="bottom">Model<br/>Link</th>
<!-- TABLE BODY -->
<tr><td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">Single</td>
<td align="center">1024</td>
<td align="center">71.3</td>
<td align="center"><a href="https://purdue0-my.sharepoint.com/:u:/g/personal/wang3702_purdue_edu/EcqhVUwoVOVJvxShvDXwF9oBs1xqrXe0_Y23NZ7guiGMlQ?e=FzmRzG">model</a></td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">Single</td>
<td align="center">4096</td>
<td align="center">72.0</td>
<td align="center"><a href="https://purdue0-my.sharepoint.com/:u:/g/personal/wang3702_purdue_edu/ETuGL8BXTz9PqZ23NGfyDJIBBQZcj38CVCxkpEC-FDOjBw?e=iTRMeW">model</a></td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">800</td>
<td align="center">Single</td>
<td align="center">4096</td>
<td align="center">75.3</td>
<td align="center">None</td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">800</td>
<td align="center">Multi</td>
<td align="center">2048</td>
<td align="center">75.7</td>
<td align="center"><a href="https://purdue0-my.sharepoint.com/:u:/g/personal/wang3702_purdue_edu/EZp5gwlcV4xFn6ir0XVsjFMBzAYIpsOI_AGvNRKeRfCtUw?e=sLpDni">model</a></td>
</tr>
</tbody></table>


## Citation:
[CaCo: Both Positive and Negative Samples are Directly Learnable via Cooperative-adversarial Contrastive Learning ]().  
```
@article{wang2022caco,
  title={CaCo: Both Positive and Negative Samples are Directly Learnable via Cooperative-adversarial Contrastive Learning },
  author={Wang, Xiao and Huang, Yuhang and Zeng, Dan and Qi, Guo-Jun},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (submitted)},
  year={2022}
}
```

