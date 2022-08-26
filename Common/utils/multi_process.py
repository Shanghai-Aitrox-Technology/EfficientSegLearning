
import traceback
from typing import Callable, Sequence, Any
from multiprocessing import Pool, cpu_count


def apply_async_multi_process(process_func: Callable, data_list: Sequence[Any], pool_ratio=0.3, pool_num=10):
    pool = Pool(min(max(pool_num, int(cpu_count() * pool_ratio)), int(cpu_count()*0.7)))
    for data in data_list:
        try:
            pool.apply_async(process_func, (data))
        except Exception as err:
            traceback.print_exc()
            print(f'Apply {process_func.__name__} throws exception: {err}!')
    pool.close()
    pool.join()