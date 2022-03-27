import os
import resource
import torch
import warnings
import random
import torch.backends.cudnn as cudnn

def Config_Environment(args):
    # increase the limit of resources to make sure it can run under any conditions
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    # config gpu settings
    use_cuda = torch.cuda.is_available()
    print("Cuda status ", use_cuda)
    ngpus_per_node = torch.cuda.device_count()
    print("in total we have ", ngpus_per_node, " gpu")
    if ngpus_per_node <= 0:
        print("We do not have gpu supporting, exit!!!")
        exit()
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    print("world size: ",args.world_size)
    #init random seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    return ngpus_per_node

def Config_hvd_Environment(args):
    # increase the limit of resources to make sure it can run under any conditions
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    import horovod.torch as hvd
    hvd.init()
    
    torch.cuda.set_device(hvd.local_rank())
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        torch.cuda.manual_seed(args.seed)
    ngpus_per_node = torch.cuda.device_count()
    #nodes_num = args.nodes_num
    args.rank = hvd.rank()
    torch.set_num_threads(4)
    args.world_size = hvd.size()#args.nodes_num*ngpus_per_node#hvd.size()#nodes_num*ngpus_per_node
    args.distributed = 1
    print("total gpus:",ngpus_per_node)
    print("world size :",args.world_size)
    print("rank: ",args.rank)
    print("hvd size:",hvd.size())