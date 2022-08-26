#! /usr/bin/python3

"""
segmentation of flare in coarse resolution.
"""

import os
import sys
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from BaseSeg.engine.runner import SegTrainRunner

from Common.utils.config import Config
from Common.utils.env import set_gpu, set_random_seed


if __name__ == '__main__':
    config_path = './full_config.yaml'
    config = Config.fromfile(config_path)

    set_gpu(config.env.num_gpu, used_percent=0.3, local_rank=0)
    set_random_seed(0, deterministic=True)

    trainer = SegTrainRunner(config)
    trainer.run()