
import os
import random
import numpy as np

import torch
import pynvml

pynvml.nvmlInit()


def set_gpu(num_gpu, used_percent=0.7, local_rank=0):
    pynvml.nvmlInit()
    print("Found %d GPU(s)" % pynvml.nvmlDeviceGetCount())

    available_gpus = []
    for index in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = meminfo.used / meminfo.total
        if used < used_percent and index >= local_rank:
            available_gpus.append(index)

    if len(available_gpus) >= num_gpu:
        gpus = ','.join(str(e) for e in available_gpus[:num_gpu])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        print("Using GPU %s" % gpus)
    else:
        raise ValueError("No GPUs available, current number of available GPU is %d, requested for %d GPU(s)" % (
            len(available_gpus), num_gpu))


def set_random_seed(seed, deterministic=False):
    """Set random seed."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True