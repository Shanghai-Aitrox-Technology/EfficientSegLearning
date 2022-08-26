#!/usr/bin/python3

import os
import sys
import argparse
import warnings

warnings.filterwarnings('ignore')


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)


from Common.utils.config import Config
from FlareSemiSeg.deploy_model.infer_model import FlareInfer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='full functional execute script of flare seg module.')
    parser.add_argument('-c', '--config', type=str, default='./config.yaml', help='config file path')
    parser.add_argument('-i', '--input_path', type=str, default='/workspace/inputs/', help='input path')
    parser.add_argument('-o', '--output_path', type=str, default='/workspace/outputs/', help='output path')
    parser.add_argument('-s', '--series_uid_path', type=str, default='', help='series uid')

    args = parser.parse_args()
    test_config = Config.fromfile(args.config)
    test_config.data_loader.test_image_dir = args.input_path
    test_config.testing.save_dir = args.output_path
    if args.series_uid_path is not None and os.path.exists(args.series_uid_path):
        test_config.data_loader.test_series_uid_txt = args.series_uid_path

    pipeline_infer = FlareInfer(test_config)
    pipeline_infer.run()