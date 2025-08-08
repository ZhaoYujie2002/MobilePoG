import os
import logging
import torch
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def init_distributed_mode(args):
    """
    Setting for distributed data parallel
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.world_size = 1
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.rank)
    dist.barrier()

def cleanup_process():
    dist.destroy_process_group()


class MultiProcessMeter(object):
    """
    Class for meter between multi-process
    """
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    def get_avg(self):
        return self.total / (self.count + 1e-9)
    def get_count(self):
        return self.count
    
def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds"

def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x

def set_logger(log_name, log_path):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(filename=log_path, mode='w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

