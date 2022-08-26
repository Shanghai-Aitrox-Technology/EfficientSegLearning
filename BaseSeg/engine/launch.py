
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

__all__ = ["DEFAULT_TIMEOUT", "launch"]

DEFAULT_TIMEOUT = timedelta(minutes=30)


def launch(
    main_func,
    world_size,
    is_distributed=True,
    dist_url=None,
    args=(),
    timeout=DEFAULT_TIMEOUT,
):
    """
    Launch multi-gpu or distributed training.
    It will spawn child processes (defined by ``num_gpus_per_machine``) on each machine.

    Args:
        main_func: a function that will be called by `main_func(*args)`
        world_size (int): number of GPUs per machine
        is_distributed(str): whether is distributed training
        dist_url (str): url to connect to for distributed jobs, including protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to "auto" to automatically select a free port on localhost
        timeout (timedelta): timeout of the distributed workers
        args (tuple): arguments passed to main_func
    """

    if world_size > 1 and is_distributed and torch.cuda.is_available():
        mp.spawn(
            _distributed_worker,
            nprocs=world_size,
            args=(
                main_func,
                world_size,
                dist_url,
                args,
                timeout,
            ),
            daemon=False,
        )
    else:
        main_func(*args)


def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    dist_url,
    args,
    timeout=DEFAULT_TIMEOUT,
):
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    try:
        dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=local_rank,
            timeout=timeout,
        )
    except Exception as e:
        raise e

    torch.cuda.set_device(local_rank)

    main_func(*args)
