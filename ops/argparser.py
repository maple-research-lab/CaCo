import parser
import argparse
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def argparser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', default="data", type=str, metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--log_path', type=str, default="train_log", help="log path for saving models")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        # choices=model_names,
                        type=str,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet50)')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning_rate', default=0.03, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr_final', default=0.0003, type=float,
                        help='final learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--mem_momentum', default=0.9, type=float, metavar='M',
                        help='memory bank momentum update in its SGD')                  
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--mem_wd',default=1e-4, type=float,
                        help='memory bank weight decay (default: 1e-4)')
    parser.add_argument('-p', '--print_freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of nodes for distributed training,args.nodes_num*args.ngpu,here we specify with the number of nodes')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training,rank of total threads, 0 to args.world_size-1')
    parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing_distributed', type=int, default=1,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    # moco specific configs:
    parser.add_argument('--moco_dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco_m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument("--moco_m_decay",default=0,type=int,help="decay network momentum or not")
    parser.add_argument('--moco_t', default=0.12, type=float,
                        help='softmax temperature (default: 0.12)')
    parser.add_argument('--mem_t', default=0.02, type=float,
                        help='temperature for memory bank(default: 0.07)')
    parser.add_argument("--mlp_dim",default=2048,type=int,help="mlp dim in projector(default:2048)")

    # options for moco v2
    parser.add_argument('--dataset', type=str, default="ImageNet", help="Specify dataset: ImageNet or cifar10")
    
    
    parser.add_argument('--cluster', type=int, default=65536, help="number of learnable comparison features")
    parser.add_argument('--memory_lr', type=float, default=3, help="learning rate for adversial memory bank")
    parser.add_argument("--memory_lr_final",type=float,default=0.03,help="memory lr final, to increase cosine schedule")
    parser.add_argument("--ad_init", type=int, default=1, help="use feature encoding to init or not")
    parser.add_argument("--nodes_num", type=int, default=1, help="number of nodes to use")
    

    parser.add_argument("--knn_freq", type=int, default=1, help="report current accuracy under specific iterations")
    parser.add_argument("--knn_batch_size", type=int, default=128, help="default batch size for knn eval")
    parser.add_argument("--knn_neighbor", type=int, default=20, help="nearest neighbor used to decide the labels")

    parser.add_argument("--type",default=0,type=int, help="running type for adcov2, try different type ideas")
    
    parser.add_argument('--warmup_epochs', default=0, type=int, metavar='N',
                    help='number of warmup epochs')
    parser.add_argument("--multicrop",default=0,type=int,help="use multicrop or not")

    parser.add_argument("--nmb_crops", type=int, default=[1, 6], nargs="+",
                        help="list of number of crops (example: [1, 6])")  # when use 0 denotes the multi crop is not applied
    parser.add_argument("--size_crops", type=int, default=[224, 96], nargs="+",
                        help="crops resolutions (example: [224, 96])")
    parser.add_argument("--min_scale_crops", type=float, default=[0.14, 0.05], nargs="+",
                        help="argument in RandomResizedCrop (example: [0.14, 0.05])")
    parser.add_argument("--max_scale_crops", type=float, default=[1.0, 0.14], nargs="+",
                        help="argument in RandomResizedCrop (example: [1., 0.14])")
    return parser