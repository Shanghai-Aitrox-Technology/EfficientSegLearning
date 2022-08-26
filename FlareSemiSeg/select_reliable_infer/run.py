#!/usr/bin/python3

import os
import sys
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from Common.utils.config import Config
from FlareSemiSeg.select_reliable_infer.model import FlareInfer

if __name__ == '__main__':
    config_path = './config.yaml'
    test_config = Config.fromfile(config_path)
    test_config.env.rank = '0'
    if not os.path.exists(test_config.testing.save_reliable_path):
        os.makedirs(test_config.testing.save_reliable_path)
    test_config.testing.save_reliable_path += 'reliable_score.csv'

    os.environ['CUDA_VISIBLE_DEVICES'] = str(test_config.env.rank)

    pipeline_infer = FlareInfer(test_config)
    pipeline_infer.run()
