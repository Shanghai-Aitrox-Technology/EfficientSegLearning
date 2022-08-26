
import os
import json
import numpy as np

import lmdb
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

from BaseSeg.data.transforms.dictionary import CenterCropD, RandomCropD,\
     RandomRotFlipD, CreateOnehotLabel, NormalizationD, RandomNoiseD, \
     RandomContrastD, RandomBrightnessAdditiveD, NormalizationMinMaxD, ResizeD

from BaseSeg.data.transforms.transform import ColorJitter, NoiseJitter, PaintingJitter
from Common.fileio.file_utils import read_txt, split_filename


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


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class SemiSegDataset(Dataset):
    def __init__(self, cfg, phase):
        self.cfg = cfg
        self.phase = phase
        self.num_class = cfg.model.num_class
        self.stage = cfg.data_loader.stage
        self.data_augment = cfg.data_augment.enable
        self.patch_size = cfg.model.input_size
        self.batch_pseudo_ratio = cfg.data_loader.batch_pseudo_ratio
        self.window_level = self.cfg.data_loader.window_level
        self.labeled_batch = cfg.data_loader.labeled_batch

        if phase != 'train' or not self.data_augment:
            self.dict_weak_transform = transforms.Compose([
                    CenterCropD(self.patch_size),
                    NormalizationD(self.window_level),
                    CreateOnehotLabel(self.num_class)])
            self.dict_strong_transform = transforms.Compose([
                    CenterCropD(self.patch_size),
                    NormalizationD(self.window_level),
                    CreateOnehotLabel(self.num_class)])
        else:
            self.dict_weak_transform = transforms.Compose([
                RandomRotFlipD(),
                ResizeD(scales=(0.8, 1.2), num_class=self.num_class),
                RandomCropD(self.patch_size),
                RandomBrightnessAdditiveD(labels=[i for i in range(1, self.num_class + 1)],
                                          additive_range=(-200, 200)),
                NormalizationD(self.window_level),
                CreateOnehotLabel(self.num_class)])
            self.dict_strong_transform = transforms.Compose([
                RandomRotFlipD(),
                ResizeD(scales=(0.8, 1.2), num_class=self.num_class),
                RandomCropD(self.patch_size),
                RandomBrightnessAdditiveD(labels=[i for i in range(1, self.num_class+1)],
                                          additive_range=(-200, 200)),
                NormalizationMinMaxD(self.window_level),
                ColorJitter(),
                NoiseJitter(),
                PaintingJitter(),
                NormalizationD([-1, 1]),
                CreateOnehotLabel(self.num_class)])

        self.pseudo_list = []
        if self.phase == 'train':
            db_file = self.cfg.data_loader.train_db
            self.labeled_list = self._read_db(db_file)
            if self.cfg.data_loader.train_aux_db is not None and \
                    os.path.exists(self.cfg.data_loader.train_aux_db):
                db_file = self.cfg.data_loader.train_aux_db
                self.pseudo_list = self._read_db(db_file)
        else:
            db_file = self.cfg.data_loader.val_db
            self.labeled_list = self._read_db(db_file)

        self.num_labeled = len(self.labeled_list)
        self.sample_list = []
        self.pseudo_idxs = []
        if len(self.pseudo_list) and phase != 'refine':
            # self.num_pseudo = len(self.pseudo_list)
            # start_idx = 0
            # for i in range(self.num_pseudo):
            #     if i % self.batch_pseudo_ratio == 0:
            #         end_idx = min(start_idx + 1, self.num_labeled)
            #         self.sample_list.extend(self.labeled_list[start_idx:end_idx])
            #         start_idx = (start_idx + 1) % (self.num_labeled-1)
            #     self.sample_list.append(self.pseudo_list[i])
            #     self.pseudo_idxs.append(self.pseudo_list[i].series_id)
            self.num_pseudo = len(self.pseudo_list)
            start_idx = 0
            for i in range(self.num_pseudo):
                end_idx = min(start_idx + self.labeled_batch, self.num_labeled)
                self.sample_list.extend(self.labeled_list[start_idx:end_idx])
                start_idx = (start_idx + self.labeled_batch) % (self.num_labeled-1)
                self.sample_list.append(self.pseudo_list[i])
                self.pseudo_idxs.append(self.pseudo_list[i].series_id)
        else:
            self.sample_list = self.labeled_list

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        """
        Return:
            image(torch tensor): channel first, dims=[c, z, y, x]
            mask(torch tensor): channel first, dims=[c, z, y, x]
        """
        data = self.sample_list[idx]
        series_id = data.series_id
        if self.stage == 1:
            image_path = data.coarse_image_path
            mask_path = data.coarse_mask_path
        elif self.stage == 2:
            image_path = data.fine_image_path
            mask_path = data.fine_mask_path
        elif self.stage == 3:
            image_path = data.part_image_path
            mask_path = data.part_mask_path

        suffix = split_filename(mask_path)[1]
        if suffix == '.npy':
            npy_mask = np.load(mask_path)
        elif suffix == '.npz':
            data_dict = np.load(mask_path)
            npy_mask = data_dict['mask']
        else:
            npy_mask = np.load(mask_path + '.npy')
        npy_image = np.load(image_path + '.npy')
        if self.num_class == 1 and self.stage == 1:
            npy_mask[npy_mask != 0] = 1

        data = {'image': npy_image, 'label': npy_mask}
        if len(self.pseudo_idxs) and series_id in self.pseudo_idxs:
            data = self.dict_strong_transform(data)
        else:
            data = self.dict_weak_transform(data)
        npy_image, mask_czyx = data['image'], data['onehot_label']
        image_czyx = npy_image[np.newaxis, ]

        return torch.from_numpy(image_czyx).float(), torch.from_numpy(mask_czyx).float()

    def _read_db(self, db_file):
        local_data = []
        env = lmdb.open(db_file, map_size=int(1e9))
        txn = env.begin()
        for key, value in txn.cursor():
            key = str(key, encoding='utf-8')
            value = str(value, encoding='utf-8')

            label_info = json.loads(value)
            tmp_data = MaskData(series_id=key,
                                image_path=label_info['image_path'],
                                mask_path=label_info['mask_path'],
                                coarse_image_path=label_info['coarse_image_path']
                                if 'coarse_image_path' in label_info else None,
                                coarse_mask_path=label_info['coarse_mask_path']
                                if 'coarse_mask_path' in label_info else None,
                                fine_image_path=label_info['fine_image_path']
                                if 'fine_image_path' in label_info else None,
                                fine_mask_path=label_info['fine_mask_path']
                                if 'fine_mask_path' in label_info else None,
                                part_image_path=label_info['part_image_path']
                                if 'part_image_path' in label_info else None,
                                part_mask_path=label_info['part_mask_path']
                                if 'part_mask_path' in label_info else None)
            local_data.append(tmp_data)
        env.close()

        return local_data


class SegDataSet(Dataset):

    def __init__(self, cfg, phase):
        super(SegDataSet, self).__init__()

        self.cfg = cfg
        self.phase = phase
        self.num_class = self.cfg.model.num_class

        if self.phase == 'train':
            self.db_file = self.cfg.data_loader.train_db
        else:
            self.db_file = self.cfg.data_loader.val_db

        self.label = cfg.data_loader.label_index
        self.stage = cfg.data_loader.stage
        self.data_augment = cfg.data_augment.enable
        self.window_level = cfg.data_loader.window_level

        self.data_info = self._read_db()

        if phase != 'train' or not self.data_augment:
            self.dict_transform = transforms.Compose([
                CenterCropD(self.cfg.model.input_size),
                NormalizationD(self.window_level),
                CreateOnehotLabel(self.num_class)])
        else:
            self.dict_transform = transforms.Compose([
                RandomRotFlipD(),
                ResizeD(scales=(0.8, 1.2), num_class=self.num_class),
                RandomCropD(self.cfg.model.input_size),
                RandomBrightnessAdditiveD(labels=[i for i in range(1, self.num_class+1)],
                                          additive_range=(-200, 200)),
                NormalizationMinMaxD(self.window_level),
                ColorJitter(),
                NoiseJitter(),
                NormalizationD([-1, 1]),
                CreateOnehotLabel(self.num_class)])

    def __len__(self):
        if self.cfg.env.smoke_test:
            return len(self.data_info[:40])
        else:
            return len(self.data_info)

    def __getitem__(self, idx):
        """
        Return:
            image(torch tensor): channel first, dims=[c, z, y, x]
            mask(torch tensor): channel first, dims=[c, z, y, x]
        """
        data = self.data_info[idx]
        if self.stage == 1:
            image_path = data.coarse_image_path
            mask_path = data.coarse_mask_path
        elif self.stage == 2:
            image_path = data.fine_image_path
            mask_path = data.fine_mask_path
        elif self.stage == 3:
            image_path = data.part_image_path
            mask_path = data.part_mask_path

        suffix = split_filename(mask_path)[1]
        if suffix == '.npy':
            npy_mask = np.load(mask_path)
        elif suffix == '.npz':
            data_dict = np.load(mask_path)
            npy_mask = data_dict['mask']
        else:
            npy_mask = np.load(mask_path + '.npy')
        npy_image = np.load(image_path + '.npy')
        if self.num_class == 1 and self.stage == 1:
            npy_mask[npy_mask != 0] = 1

        data = {'image': npy_image, 'label': npy_mask}
        data = self.dict_transform(data)
        npy_image, mask_czyx = data['image'], data['onehot_label']
        image_czyx = npy_image[np.newaxis, ]

        return torch.from_numpy(image_czyx).float(), torch.from_numpy(mask_czyx).float()

    def _read_db(self):
        bad_case_path = self.cfg.data_loader.bad_case_file
        if self.phase == 'train' and bad_case_path is not None and os.path.exists(bad_case_path):
            bad_case_uid = read_txt(bad_case_path)
        else:
            bad_case_uid = None
        local_data = []
        env = lmdb.open(self.db_file, map_size=int(1e9))
        txn = env.begin()
        for key, value in txn.cursor():
            key = str(key, encoding='utf-8')
            value = str(value, encoding='utf-8')

            label_info = json.loads(value)
            tmp_data = MaskData(series_id=key,
                                image_path=label_info['image_path'],
                                mask_path=label_info['mask_path'],
                                coarse_image_path=label_info['coarse_image_path']
                                if 'coarse_image_path' in label_info else None,
                                coarse_mask_path=label_info['coarse_mask_path']
                                if 'coarse_mask_path' in label_info else None,
                                fine_image_path=label_info['fine_image_path']
                                if 'fine_image_path' in label_info else None,
                                fine_mask_path=label_info['fine_mask_path']
                                if 'fine_mask_path' in label_info else None,
                                part_image_path=label_info['part_image_path']
                                if 'part_image_path' in label_info else None,
                                part_mask_path=label_info['part_mask_path']
                                if 'part_mask_path' in label_info else None)
            if bad_case_uid is not None and tmp_data.series_id in bad_case_uid:
                local_data.extend([tmp_data]*self.cfg.data_loader.bad_case_augment_times)

            local_data.append(tmp_data)
        env.close()

        return local_data