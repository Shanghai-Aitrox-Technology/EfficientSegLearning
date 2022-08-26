# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import time
import pynvml
import inspect
import datetime
from thop import profile
from functools import wraps
from pynvml.smi import nvidia_smi
import numpy as np

import torch
from typing import Optional

from Common.fileio.file_utils import write_txt

__all__ = ["timer_decorate", "torch_profiler_full", "torch_profiler_time_cpu_gpu",
           "torch_profiler_time_end_to_end", "PerfContext"]


def get_time(use_cuda=True):
    if use_cuda:
        torch.cuda.synchronize()
    return time.time()


def get_gpu_memory_usage(time_interval: int, is_ongoing_monitor: bool = True,
                         gpu_index: int = 0, out_path: Optional[str] = None):
        gpu_memory_max = 0
        interval_nums = 0
        while is_ongoing_monitor:
            interval_nums += 1
            nvsmi = nvidia_smi.getInstance()
            dictm = nvsmi.DeviceQuery('memory.free, memory.total')
            gpu_memory = dictm['gpu'][int(gpu_index)]['fb_memory_usage']['total'] - \
                         dictm['gpu'][int(gpu_index)]['fb_memory_usage']['free']
            # gpu_memory_total = dictm['gpu'][int(gpu_index)]['fb_memory_usage']['total']
            gpu_memory_max = max(gpu_memory_max, gpu_memory)
            if is_ongoing_monitor:
                time.sleep(time_interval)
                if interval_nums % 20 == 0:
                    if out_path is not None:
                        write_txt(out_path, [str(gpu_memory_max)])

        return gpu_memory_max


def gpu_profiler(gpu_index: int = 0, time_sleep: float = 0.1):
    time.sleep(time_sleep)
    torch.cuda.synchronize()
    nvsmi = nvidia_smi.getInstance()
    dictm = nvsmi.DeviceQuery('memory.free, memory.total')
    gpu_memory = dictm['gpu'][int(gpu_index)]['fb_memory_usage']['total'] - \
                 dictm['gpu'][int(gpu_index)]['fb_memory_usage']['free']

    return gpu_memory


def get_params_flops(net, input_array):
    flops, params = profile(net, inputs=(input_array))

    return flops, params


def timer_decorate(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        t_start = time.time()

        result = func(*args, **kwargs)

        torch.cuda.synchronize()
        t_end = time.time()
        time_consume = t_end - t_start

        return result, time_consume

    return wrapper


def torch_profiler_full(func):
    """
    A decorator which will run the torch profiler for the decorated function,
    printing the results in full.
    Note: Enforces a gpu sync point which could slow down pipelines.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            result = func(*args, **kwargs)

        print(prof, flush=True)

        return result

    return wrapper


def torch_profiler_time_cpu_gpu(func):
    """
    A decorator which measures the execution time of both the CPU and GPU components
    of the decorated function, printing both results.
    Note: Enforces a gpu sync point which could slow down pipelines.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            result = func(*args, **kwargs)

        cpu_time = prof.self_cpu_time_total
        gpu_time = sum(evt.self_cuda_time_total for evt in prof.function_events)

        cpu_time = torch.autograd.profiler.format_time(cpu_time)
        gpu_time = torch.autograd.profiler.format_time(gpu_time)

        print(f"cpu time: {cpu_time}, gpu time: {gpu_time}", flush=True)

        return result

    return wrapper


def torch_profiler_time_end_to_end(func):
    """
    A decorator which measures the total execution time from when the decorated
    function is called to when the last cuda operation finishes, printing the result.
    Note: Enforces a gpu sync point which could slow down pipelines.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        torch.cuda.synchronize()
        start = time.perf_counter()

        result = func(*args, **kwargs)

        torch.cuda.synchronize()
        end = time.perf_counter()

        total_time = (end - start) * 1e6
        total_time_str = torch.autograd.profiler.format_time(total_time)
        print(f"end to end time: {total_time_str}", flush=True)

        return result

    return wrapper


class PerfContext:
    """
    Context manager for tracking how much time is spent within context blocks. This uses `time.perf_counter` to
    accumulate the total amount of time in seconds in the attribute `total_time` over however many context blocks
    the object is used in.
    """

    def __init__(self):
        self.total_time = 0
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.total_time += time.perf_counter() - self.start_time
        self.start_time = None


def memory_decorate(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        frame = inspect.currentframe()  # define a frame to track
        gpu_tracker = MemTracker(frame, path=f'{func.__name__}_gpu_mem_track.txt')  # define a GPU tracker
        gpu_tracker.track()
        result = func(*args, **kwargs)
        gpu_tracker.track()

        return result

    return wrapper


class MemTracker(object):
    """
    Class used to track pytorch memory usage
    Arguments:
        frame: a frame to detect current py-file runtime
        detail(bool, default True): whether the function shows the detail gpu memory usage
        path(str): where to save log file
        verbose(bool, default False): whether show the trivial exception
        device(int): GPU number, default is 0

    Examples:
        frame = inspect.currentframe()  # define a frame to track
        gpu_tracker = MemTracker(frame)  # define a GPU tracker
        gpu_tracker.track()
        ---------------------------
        test code
        ---------------------------
        gpu_tracker.track()

    """
    def __init__(self, frame, detail=True, path='', verbose=False, device=0):
        self.frame = frame
        self.print_detail = detail
        self.last_tensor_sizes = set()
        self.gpu_profile_fn = path + f'{datetime.datetime.now():%d-%b-%y-%H:%M:%S}-gpu_mem_track.txt' \
            if path == '' else path
        self.verbose = verbose
        self.begin = True
        self.device = device

        self.func_name = frame.f_code.co_name
        self.filename = frame.f_globals["__file__"]
        if (self.filename.endswith(".pyc") or
                self.filename.endswith(".pyo")):
            self.filename = self.filename[:-1]
        self.module_name = self.frame.f_globals["__name__"]
        self.curr_line = self.frame.f_lineno

    def get_tensors(self):
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    tensor = obj
                else:
                    continue
                if tensor.is_cuda:
                    yield tensor
            except Exception as e:
                if self.verbose:
                    print('A trivial exception occured: {}'.format(e))

    def track(self):
        """
        Track the GPU memory usage
        """
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        self.curr_line = self.frame.f_lineno
        where_str = self.module_name + ' ' + self.func_name + ':' + ' line ' + str(self.curr_line)

        with open(self.gpu_profile_fn, 'a+') as f:

            if self.begin:
                f.write(f"GPU Memory Track | {datetime.datetime.now():%d-%b-%y-%H:%M:%S} |"
                        f" Total Used Memory:{meminfo.used/1000**2:<7.1f}Mb\n\n")
                self.begin = False

            if self.print_detail is True:
                ts_list = [tensor.size() for tensor in self.get_tensors()]
                new_tensor_sizes = {(type(x), tuple(x.size()), ts_list.count(x.size()), np.prod(np.array(x.size()))*4/1000**2)
                                    for x in self.get_tensors()}
                for t, s, n, m in new_tensor_sizes - self.last_tensor_sizes:
                    f.write(f'+ | {str(n)} * Size:{str(s):<20} | Memory: {str(m*n)[:6]} M | {str(t):<20}\n')
                for t, s, n, m in self.last_tensor_sizes - new_tensor_sizes:
                    f.write(f'- | {str(n)} * Size:{str(s):<20} | Memory: {str(m*n)[:6]} M | {str(t):<20} \n')
                self.last_tensor_sizes = new_tensor_sizes

            f.write(f"\nAt {where_str:<50}"
                    f"Total Used Memory:{meminfo.used/1000**2:<7.1f}Mb\n\n")

        pynvml.nvmlShutdown()


def cumulative_add():
    res = 0
    for i in range(10):
        res += i
    a_tensor = torch.tensor(0).float().cuda()
    for i in range(10):
        a_tensor += 1
    print(res)
    print(a_tensor)


if __name__ == '__main__':
    func = torch_profiler_time_cpu_gpu(cumulative_add)
    func()
