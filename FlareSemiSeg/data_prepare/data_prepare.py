
import os
import sys
import json
import traceback

import lmdb
import numpy as np
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from Common.transforms.image_io import load_sitk_image
from Common.fileio.file_utils import MyEncoder, read_txt, write_txt
from Common.transforms.image_resample import ScipyResample
from Common.transforms.mask_process import crop_image_according_to_mask


def run_prepare_data(cfg):
    data_prepare = DataPrepare(cfg)
    async_multi_process(data_prepare.process, data_prepare.train_data_info, phase='train', pool_ratio=0.8)
    if data_prepare.val_data_info is not None:
        async_multi_process(data_prepare.process, data_prepare.val_data_info, phase='val', pool_ratio=0.8)


def async_multi_process(process_func, data_list, phase='train', pool_ratio=0.3, pool_num=10):
    print(f'Apply async multi process in {phase} phase...')
    pool = Pool(min(max(pool_num, int(cpu_count() * pool_ratio)), int(cpu_count()*0.8)))
    for data in data_list:
        try:
            pool.apply_async(process_func, (data, phase))
        except Exception as err:
            traceback.print_exc()
            print(f'Create coarse/fine image/mask throws exception: {err}, with series_id: {data.series_id}!')
    pool.close()
    pool.join()
    print(f'{phase} multi process finish!')


class MaskData(object):
    def __init__(self, series_id, image_path, mask_path,
                 smooth_mask_path=None, coarse_image_path=None, coarse_mask_path=None,
                 fine_image_path=None, fine_mask_path=None,
                 part_image_path=None, part_mask_path=None,
                 coarse_image_name=None, coarse_mask_name=None,
                 fine_image_name=None, fine_mask_name=None,
                 part_image_name=None, part_mask_name=None):
        super(MaskData, self).__init__()

        self.series_id = series_id
        self.image_path = image_path
        self.mask_path = mask_path
        self.smooth_mask_path = smooth_mask_path
        self.coarse_image_path = coarse_image_path
        self.coarse_mask_path = coarse_mask_path
        self.fine_image_path = fine_image_path
        self.fine_mask_path = fine_mask_path
        self.part_image_path = part_image_path
        self.part_mask_path = part_mask_path
        self.coarse_image_name = coarse_image_name
        self.coarse_mask_name = coarse_mask_name
        self.fine_image_name = fine_image_name
        self.fine_mask_name = fine_mask_name
        self.part_image_name = part_image_name
        self.part_mask_name = part_mask_name


class DataPrepare(object):
    def __init__(self, cfg):
        super(DataPrepare, self).__init__()
        self.cfg = cfg
        self.out_dir = cfg.DATA_PREPARE.OUT_DIR
        self.db_dir = self.out_dir + '/db/'
        self.train_db_file = self.db_dir + 'seg_train_fold_1'
        self.val_db_file = self.db_dir + 'seg_val_fold_1'
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
        self.file_dir = self.out_dir + '/file_list/'
        if not os.path.exists(self.file_dir):
            os.makedirs(self.file_dir)

        self.image_dir = cfg.DATA_PREPARE.IMAGE_DIR
        self.mask_dir = cfg.DATA_PREPARE.MASK_DIR
        self.coarse_mask_label = cfg.DATA_PREPARE.COARSE_MASK_LABEL
        self.fine_mask_label = cfg.DATA_PREPARE.FINE_MASK_LABEL
        self.part_label = cfg.DATA_PREPARE.PART_LABEL
        self.extend_size = cfg.DATA_PREPARE.EXTEND_SIZE
        self.out_coarse_size = cfg.DATA_PREPARE.OUT_COARSE_SIZE
        self.out_fine_size = cfg.DATA_PREPARE.OUT_FINE_SIZE
        self.out_part_size = cfg.DATA_PREPARE.OUT_PART_SIZE

        self.all_series_uid = read_txt(self.cfg.DATA_PREPARE.ALL_SERIES_IDS_TXT)
        self._split_train_val()
        self._creat_data_info(phase='train')
        if self.val_series_uid is not None:
            self._creat_data_info(phase='val')
        else:
            self.val_data_info = None

        # create dir to save coarse image and mask
        coarse_prefix = '{}_{}_{}'.format(self.out_coarse_size[0],
                                          self.out_coarse_size[1],
                                          self.out_coarse_size[2])
        self.coarse_image_file = 'coarse_image/' + coarse_prefix
        self.coarse_image_save_dir = os.path.join(self.out_dir, self.coarse_image_file)
        if not os.path.exists(self.coarse_image_save_dir):
            os.makedirs(self.coarse_image_save_dir)

        self.coarse_mask_file = 'coarse_mask/' + coarse_prefix
        self.coarse_mask_save_dir = os.path.join(self.out_dir, self.coarse_mask_file)
        if not os.path.exists(self.coarse_mask_save_dir):
            os.makedirs(self.coarse_mask_save_dir)

        # create dir to save fine image and mask
        fine_prefix = '{}_{}_{}'.format(self.out_fine_size[0],
                                        self.out_fine_size[1],
                                        self.out_fine_size[2])
        self.fine_image_file = 'fine_image/' + fine_prefix
        self.fine_image_save_dir = os.path.join(self.out_dir, self.fine_image_file)
        if not os.path.exists(self.fine_image_save_dir):
            os.makedirs(self.fine_image_save_dir)

        self.fine_mask_file = 'fine_mask/' + fine_prefix
        self.fine_mask_save_dir = os.path.join(self.out_dir, self.fine_mask_file)
        if not os.path.exists(self.fine_mask_save_dir):
            os.makedirs(self.fine_mask_save_dir)

        # create dir to save part image and mask
        part_prefix = '{}_{}_{}'.format(self.out_part_size[0],
                                        self.out_part_size[1],
                                        self.out_part_size[2])
        self.part_image_file = 'part_image/' + part_prefix
        self.part_image_save_dir = os.path.join(self.out_dir, self.part_image_file)
        if not os.path.exists(self.part_image_save_dir):
            os.makedirs(self.part_image_save_dir)

        self.part_mask_file = 'part_mask/' + part_prefix
        self.part_mask_save_dir = os.path.join(self.out_dir, self.part_mask_file)
        if not os.path.exists(self.part_mask_save_dir):
            os.makedirs(self.part_mask_save_dir)

    def process(self, data, phase='train'):
        series_id = data.series_id
        image_path = data.image_path
        mask_path = data.mask_path
        print('Start processing %s.' % series_id)
        image_info = load_sitk_image(image_path)
        npy_image = image_info['npy_image']
        image_spacing = image_info['spacing']

        coarse_image, _ = ScipyResample.resample_to_size(npy_image, self.out_coarse_size)
        data.coarse_image_name = os.path.join(self.coarse_image_file, series_id)
        data.coarse_image_path = os.path.join(self.coarse_image_save_dir, series_id)
        np.save(data.coarse_image_path, coarse_image)
        if mask_path is not None:
            mask_info = load_sitk_image(mask_path)
            npy_mask = mask_info['npy_image']
            npy_mask = npy_mask.astype(np.uint8)

            multi_mask = npy_mask.copy()
            if self.coarse_mask_label == 1:
                multi_mask[multi_mask != 0] = 1

            coarse_mask, _ = ScipyResample.resample_mask_to_size(multi_mask, self.out_coarse_size,
                                                                 num_label=self.coarse_mask_label)
            data.coarse_mask_name = os.path.join(self.coarse_mask_file, series_id)
            data.coarse_mask_path = os.path.join(self.coarse_mask_save_dir, series_id)
            np.save(data.coarse_mask_path, coarse_mask)

            # Process and save fine image and mask
            extend_size = [self.extend_size] * 3
            margin = [int(extend_size[0] / image_spacing[0]),
                      int(extend_size[1] / image_spacing[1]),
                      int(extend_size[2] / image_spacing[2])]
            crop_image, crop_mask = crop_image_according_to_mask(npy_image, npy_mask, margin)
            fine_image, _ = ScipyResample.resample_to_size(crop_image, self.out_fine_size)
            fine_mask, _ = ScipyResample.resample_mask_to_size(crop_mask, self.out_fine_size,
                                                               num_label=self.fine_mask_label)

            data.fine_image_name = os.path.join(self.fine_image_file, series_id)
            data.fine_mask_name = os.path.join(self.fine_mask_file, series_id)
            data.fine_image_path = os.path.join(self.fine_image_save_dir, series_id)
            data.fine_mask_path = os.path.join(self.fine_mask_save_dir, series_id)
            np.save(data.fine_image_path, fine_image)
            np.save(data.fine_mask_path, fine_mask)

            part_mask = npy_mask.copy()
            for label in range(1, 14):
                if label not in self.part_label:
                    part_mask[part_mask == label] = 0
                else:
                    idx = self.part_label.index(label)+1
                    if idx != label:
                        part_mask[part_mask == label] = idx
            crop_image, crop_mask = crop_image_according_to_mask(npy_image, part_mask, margin)
            part_image, _ = ScipyResample.resample_to_size(crop_image, self.out_part_size)
            part_mask, _ = ScipyResample.resample_mask_to_size(crop_mask, self.out_part_size,
                                                               num_label=len(self.part_label))

            data.part_image_name = os.path.join(self.part_image_file, series_id)
            data.part_mask_name = os.path.join(self.part_mask_file, series_id)
            data.part_image_path = os.path.join(self.part_image_save_dir, series_id)
            data.part_mask_path = os.path.join(self.part_mask_save_dir, series_id)
            np.save(data.part_image_path, part_image)
            np.save(data.part_mask_path, part_mask)

        self._update_db(data, phase=phase)
        print('End processing %s.' % series_id)

    def _update_db(self, data, phase='train'):
        out_db_file = self.train_db_file if phase == 'train' else self.val_db_file
        env = lmdb.open(out_db_file, map_size=int(1e9))
        txn = env.begin(write=True)

        data_dict = {'image_path': data.image_path,
                     'mask_path': data.mask_path,
                     'smooth_mask_path': data.smooth_mask_path,
                     'coarse_image_path': data.coarse_image_path,
                     'coarse_mask_path': data.coarse_mask_path,
                     'fine_image_path': data.fine_image_path,
                     'fine_mask_path': data.fine_mask_path,
                     'part_image_path': data.part_image_path,
                     'part_mask_path': data.part_mask_path,
                     'coarse_image_name': data.coarse_image_name,
                     'coarse_mask_name': data.coarse_mask_name,
                     'fine_image_name': data.fine_image_name,
                     'fine_mask_name': data.fine_mask_name,
                     'part_image_name': data.part_image_name,
                     'part_mask_name': data.part_mask_name}

        txn.put(str(data.series_id).encode(), value=json.dumps(data_dict, cls=MyEncoder).encode())

        txn.commit()
        env.close()

    def _creat_data_info(self, phase='train'):
        all_series_uid = self.train_series_uid if phase == 'train' else self.val_series_uid
        all_data_info = []
        for series_uid in all_series_uid:
            image_path = os.path.join(self.image_dir, series_uid + self.cfg.DATA_PREPARE.IMAGE_SUFFIX)
            mask_path = os.path.join(self.mask_dir, series_uid + self.cfg.DATA_PREPARE.MASK_SUFFIX)
            if not os.path.exists(image_path):
                image_path = None
            if not os.path.exists(mask_path):
                mask_path = None
            all_data_info.append(MaskData(series_id=series_uid,
                                          image_path=image_path,
                                          mask_path=mask_path))
        if phase == 'train':
            self.train_data_info = all_data_info
        else:
            self.val_data_info = all_data_info

    def _split_train_val(self):
        if eval(self.cfg.DATA_PREPARE.TRAIN_SERIES_IDS_TXT) is not None and \
                eval(self.cfg.DATA_PREPARE.VAL_SERIES_IDS_TXT) is not None:
            self.train_series_uid = read_txt(self.cfg.DATA_PREPARE.TRAIN_SERIES_IDS_TXT)
            self.val_series_uid = read_txt(self.cfg.DATA_PREPARE.VAL_SERIES_IDS_TXT)
        else:
            if len(self.all_series_uid) == 0:
                raise ValueError('Num of series is 0!')
            if self.cfg.DATA_PREPARE.VAL_RATIO > 0:
                self.train_series_uid, self.val_series_uid = \
                    train_test_split(self.all_series_uid,
                                     test_size=self.cfg.DATA_PREPARE.VAL_RATIO,
                                     random_state=0)
                write_txt(self.file_dir + 'train_series_uid.txt', self.train_series_uid)
                write_txt(self.file_dir + 'val_series_uid.txt', self.val_series_uid)
                print(f'Num of train series is: {len(self.train_series_uid)}, '
                      f'num of val series is: {len(self.val_series_uid)}')
                if self.cfg.DATA_PREPARE.IS_SPLIT_5FOLD:
                    self._split_5fold_train_val()
            else:
                self.train_series_uid = self.all_series_uid
                self.val_series_uid = None
                write_txt(self.file_dir + 'train_series_uid.txt', self.train_series_uid)

    def _split_5fold_train_val(self):
        default_train_series_uid = self.train_series_uid
        default_val_series_uid = self.val_series_uid
        num_train = len(default_train_series_uid)
        new_train_series_uid = [default_train_series_uid[:int(num_train * 0.25)],
                                default_train_series_uid[int(num_train * 0.25):int(num_train * 0.5)],
                                default_train_series_uid[int(num_train * 0.5):int(num_train * 0.75)],
                                default_train_series_uid[int(num_train * 0.75):]]

        self.train_5fold_series_uid = []
        self.val_5fold_series_uid = []
        for i in range(4):
            out_5fold_train = []
            for j in range(4):
                if i != j:
                    out_5fold_train.extend(new_train_series_uid[j])
            out_5fold_train.extend(default_val_series_uid)
            out_5fold_val = new_train_series_uid[i]
            self.train_5fold_series_uid.append(out_5fold_train)
            self.val_5fold_series_uid.append(out_5fold_val)