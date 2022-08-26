"""
Reference to 'https://github.com/HiLab-git/SSL4MIS'
"""

import json
import lmdb
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from BaseSeg.data.transforms.array import CenterCrop, RandomCrop, RandomRotFlip, ToTensor, Normalization
from BaseSeg.data.transforms.dictionary import CenterCropD, RandomCropD,\
     RandomRotFlipD, CreateOnehotLabel, ToTensorD, NormalizationD


class SampleDataset(Dataset):
    def __init__(self, cfg, phase):
        self.cfg = cfg
        self.phase = phase
        self.num_class = cfg.model.num_class
        self.is_coarse = cfg.data_loader.is_coarse
        self.data_augment = cfg.data_augment.enable
        self.patch_size = cfg.model.input_size
        self.window_level = self.cfg.data_loader.window_level

        if phase != 'train' or not self.data_augment:
            self.array_transform = transforms.Compose([
                    CenterCrop(self.patch_size),
                    Normalization(self.window_level),
                    ToTensor()])
            self.dict_transform = transforms.Compose([
                    CenterCropD(self.patch_size),
                    NormalizationD(self.window_level),
                    CreateOnehotLabel(self.num_class),
                    ToTensorD()])
        else:
            self.array_transform = transforms.Compose([
                RandomRotFlip(),
                RandomCrop(self.patch_size),
                Normalization(self.window_level),
                ToTensor()])
            self.dict_transform = transforms.Compose([
                RandomRotFlipD(),
                RandomCropD(self.patch_size),
                NormalizationD(self.window_level),
                CreateOnehotLabel(self.num_class),
                ToTensorD()])

        self.sample_list = []
        if self.phase == 'train':
            db_file = self.cfg.data_loader.train_db
            self.sample_list = self._read_db(db_file)
            self.num_labeled = len(self.sample_list)
            if self.cfg.model.semi_supervised:
                db_file = self.cfg.data_loader.train_aux_db
                self.sample_list += self._read_db(db_file)
        else:
            db_file = self.cfg.data_loader.val_db
            self.sample_list = self._read_db(db_file)

        self.num_total = len(self.sample_list)
        print("total {} samples".format(self.num_total))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        data = self.sample_list[idx]
        if self.is_coarse:
            image_path = data.coarse_image_path
            mask_path = data.coarse_mask_path
        else:
            image_path = data.fine_image_path
            mask_path = data.fine_mask_path
        image = np.load(image_path + '.npy')
        if mask_path is not None:
            label = np.load(mask_path + '.npy')
            if self.num_class == 1 and self.is_coarse:
                label[label != 0] = 1

            sample = {'image': image, 'label': label}
            sample = self.dict_transform(sample)
            sample.pop('label')
        else:
            image = self.array_transform(image)
            label = np.zeros([self.num_class, self.patch_size[0],
                              self.patch_size[1], self.patch_size[2]])
            label = torch.from_numpy(label).float()
            sample = {'image': image, 'onehot_label': label}

        if self.phase == 'train':
            return sample
        else:
            return sample['image'], sample['onehot_label']

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
                                if 'fine_mask_path' in label_info else None)
            local_data.append(tmp_data)
        env.close()

        return local_data


class MaskData(object):
    def __init__(self, series_id, image_path, mask_path,
                 smooth_mask_path=None, coarse_image_path=None, coarse_mask_path=None,
                 fine_image_path=None, fine_mask_path=None,
                 coarse_image_name=None, coarse_mask_name=None,
                 fine_image_name=None, fine_mask_name=None):
        super(MaskData, self).__init__()

        self.series_id = series_id
        self.image_path = image_path
        self.mask_path = mask_path
        self.smooth_mask_path = smooth_mask_path
        self.coarse_image_path = coarse_image_path
        self.coarse_mask_path = coarse_mask_path
        self.fine_image_path = fine_image_path
        self.fine_mask_path = fine_mask_path
        self.coarse_image_name = coarse_image_name
        self.coarse_mask_name = coarse_mask_name
        self.fine_image_name = fine_image_name
        self.fine_mask_name = fine_mask_name
