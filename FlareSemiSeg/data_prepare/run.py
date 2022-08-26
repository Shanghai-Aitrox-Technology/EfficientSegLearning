#!/usr/bin/python3

import os
import sys
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from Common.utils.config import Config
from data_prepare import run_prepare_data


if __name__ == '__main__':
    config_path = './pseudo_config.yaml'
    cfg = Config.fromfile(config_path)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if not os.path.exists(cfg.ENVIRONMENT.DATA_BASE_DIR):
        cfg.ENVIRONMENT.DATA_BASE_DIR = os.path.join(base_dir, cfg.ENVIRONMENT.DATA_BASE_DIR)
    if cfg.DATA_PREPARE.OUT_DIR is None:
        cfg.DATA_PREPARE.OUT_DIR = cfg.ENVIRONMENT.DATA_BASE_DIR

    if cfg.ENVIRONMENT.DATA_BASE_DIR is not None:
        if cfg.DATA_PREPARE.ALL_SERIES_IDS_TXT is not None and \
                not os.path.exists(cfg.DATA_PREPARE.ALL_SERIES_IDS_TXT):
            cfg.DATA_PREPARE.ALL_SERIES_IDS_TXT = cfg.ENVIRONMENT.DATA_BASE_DIR + \
                                                  cfg.DATA_PREPARE.ALL_SERIES_IDS_TXT
        if cfg.DATA_PREPARE.IMAGE_DIR is not None and not os.path.exists(cfg.DATA_PREPARE.IMAGE_DIR):
            cfg.DATA_PREPARE.IMAGE_DIR = cfg.ENVIRONMENT.DATA_BASE_DIR + \
                                         cfg.DATA_PREPARE.IMAGE_DIR
        if cfg.DATA_PREPARE.MASK_DIR is not None and \
                not os.path.exists(cfg.DATA_PREPARE.MASK_DIR):
            cfg.DATA_PREPARE.MASK_DIR = cfg.ENVIRONMENT.DATA_BASE_DIR + \
                                        cfg.DATA_PREPARE.MASK_DIR
        if eval(cfg.DATA_PREPARE.TRAIN_SERIES_IDS_TXT) is not None and \
                not os.path.exists(cfg.DATA_PREPARE.TRAIN_SERIES_IDS_TXT):
            cfg.DATA_PREPARE.TRAIN_SERIES_IDS_TXT = cfg.ENVIRONMENT.DATA_BASE_DIR + \
                                                    cfg.DATA_PREPARE.TRAIN_SERIES_IDS_TXT
        if eval(cfg.DATA_PREPARE.VAL_SERIES_IDS_TXT) is not None and \
                not os.path.exists(cfg.DATA_PREPARE.VAL_SERIES_IDS_TXT):
            cfg.DATA_PREPARE.VAL_SERIES_IDS_TXT = cfg.ENVIRONMENT.DATA_BASE_DIR + \
                                                  cfg.DATA_PREPARE.VAL_SERIES_IDS_TXT

    run_prepare_data(cfg)
