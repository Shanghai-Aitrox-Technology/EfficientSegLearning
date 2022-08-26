
import copy
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from typing import List, Optional

from Common.transforms.image_io import load_sitk_image
from BaseSeg.data.transforms.dictionary import CenterCropD, RandomCropD,\
     RandomRotFlipD, NormalizationD, ResizeD, RandomNoiseD, \
     RandomContrastD, RandomBrightnessAdditiveD, NormalizationMinMaxD

from BaseSeg.data.transforms.transform import ColorJitter, NoiseJitter, PaintingJitter

from Common.transforms.image_resample import ScipyResample
from Common.transforms.mask_process import crop_image_according_to_mask


def plot_slice_image_mask(image, mask, title, out_dir):
    zz, yy, xx = np.where(mask != 0)
    x_mid_slice = int(np.median(xx))

    image_slice = image[:, :, x_mid_slice]
    mask_slice = mask[:, :, x_mid_slice]
    masked = np.ma.masked_where(mask_slice == 0, mask_slice)
    out_path = out_dir + title + '.jpg'

    plt.figure()
    plt.imshow(image_slice, cmap='gray')
    plt.imshow(masked, cmap='jet', alpha=0.1)
    plt.title(title)
    plt.show()
    plt.savefig(out_path)


class RandomGamma(object):
    def __init__(self, gamma_range=(0, 0.5), p=0.5):
        self.p = p
        self.gamma_range = gamma_range

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample
        image, label = sample['image'], sample['label']
        if np.random.random() < 0.5 and self.gamma_range[0] < 1:
            gamma = np.random.uniform(self.gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(self.gamma_range[0], 1), self.gamma_range[1])
        minm = image.min()
        rnge = image.max() - minm
        image = np.power(((image - minm) / float(rnge + 1e-5)), gamma) * rnge + minm

        return {'image': image, 'label': label}


class ImagePaintingD(object):
    def __init__(self, p=0.5, in_painting_p=0.5, in_painting_range=(0, 0.5), out_painting_range=(0.8, 0.9)):
        self.p = p
        self.in_painting_p = in_painting_p
        self.in_painting_range = in_painting_range
        self.out_painting_range = out_painting_range

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample
        x, label = sample['image'], sample['label']
        if random.random() < self.in_painting_p:
            x = self._in_painting(x)
        else:
            x = self._out_painting(x)

        return {'image': x, 'label': label}

    def _in_painting(self, x):
        image_shape = x.shape
        cnt = 3
        while cnt > 0 and random.random() < 0.95:
            block_noise_size = [random.randint(int(item*self.in_painting_range[0]),
                                               int(item*self.in_painting_range[1])) for item in image_shape]
            noise_start = [random.randint(3, image_shape[i] - block_noise_size[i] - 3) for i in range(3)]
            x[noise_start[0]:noise_start[0] + block_noise_size[0],
              noise_start[1]:noise_start[1] + block_noise_size[1],
              noise_start[2]:noise_start[2] + block_noise_size[2]] = \
                np.random.rand(block_noise_size[0],
                               block_noise_size[1],
                               block_noise_size[2]) * 1.0
            cnt -= 1
        return x

    def _out_painting(self, x):
        image_shape = x.shape
        img_rows, img_cols, img_deps = image_shape
        image_temp = copy.deepcopy(x)
        x = np.random.rand(img_rows, img_cols, img_deps) * 1.0
        cnt = 3
        while cnt > 0 and random.random() < 0.95:
            block_noise_size = [random.randint(int(self.out_painting_range[0] * item),
                                               int(self.out_painting_range[1] * item)) for item in image_shape]
            noise_start = [random.randint(3, image_shape[i] - block_noise_size[i] - 3) for i in range(3)]
            x[noise_start[0]:noise_start[0] + block_noise_size[0],
              noise_start[1]:noise_start[1] + block_noise_size[1],
              noise_start[2]:noise_start[2] + block_noise_size[2]] = \
                image_temp[noise_start[0]:noise_start[0] + block_noise_size[0],
                           noise_start[1]:noise_start[1] + block_noise_size[1],
                           noise_start[2]:noise_start[2] + block_noise_size[2]]
            cnt -= 1

        return x


class GaussianBlurD(object):
    def __init__(self, sigma=(0, 0.5), p=0.5):
        self.p = p
        self.sigma = sigma

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample
        image, label = sample['image'], sample['label']
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        image = gaussian_filter(image, sigma, order=0)

        return {'image': image, 'label': label}


def data_augment_test(image, mask, crop_size):
    plot_slice_image_mask(image, mask, 'raw image', out_dir)
    sample = {'image': image, 'label': mask}

    # transform_dict = {
    #     'random rot flip': RandomRotFlipD(p=1),
    #     'random resize': ResizeD(scales=(0.9, 1.1), num_class=13, p=1),
    #     'random crop': RandomCropD(crop_size),
    #     # 'random brightness': RandomBrightnessAdditiveD(labels=[i for i in range(1, 14)],
    #     #                                                additive_range=(-200, 200), p=1),
    #     'clip min max': NormalizationMinMaxD([-500, 500]),
    #     # 'random noise': RandomNoiseD(noise_variance=(0, 0.2), p=1),
    #     # 'random contrast': RandomContrastD(alpha=(0.8, 1.2), mean=0.25, p=1.0),
    #     # 'random gamma': RandomGamma(gamma_range=(0, 2.0), p=1.0),
    #     # 'gaussian blur': GaussianBlurD(sigma=(0.5, 1.0), p=1.0),
    #     'image painting': ImagePaintingD(p=1.0, in_painting_p=1.0),
    #     # 'image_norm': NormalizationD([-1, 1])
    # }

    transform_dict = {
        'random rot flip': RandomRotFlipD(p=1),
        'random resize': ResizeD(scales=(0.9, 1.1), num_class=13, p=1),
        'random crop': RandomCropD(crop_size),
        'random brightness': RandomBrightnessAdditiveD(labels=[i for i in range(1, 14)],
                                                       additive_range=(-200, 200), p=1),
        'clip min max': NormalizationMinMaxD([-500, 500]),
        'color jitter': ColorJitter(p=1.0),
        'noise jitter': NoiseJitter(p=1.0),
        'painting jitter': PaintingJitter(p=1.0),
        'image_norm': NormalizationD([-1, 1])
    }

    for key, transform in transform_dict.items():
        sample = transform(sample)
        plot_slice_image_mask(sample['image'], sample['label'], key, out_dir)


if __name__ == '__main__':
    image_path = '/data/zhangfan/FLARE2022/raw_data/crop_labeled_image/FLARE22_Tr_0001.nii.gz'
    mask_path = '/data/zhangfan/FLARE2022/raw_data/crop_labeled_mask/FLARE22_Tr_0001.nii.gz'
    out_dir = '/data/zhangfan/FLARE2022/results/transform_image/'

    # image = np.load(image_path)
    # mask = np.load(mask_path)
    image_dict = load_sitk_image(image_path)
    mask_dict = load_sitk_image(mask_path)
    image = image_dict['npy_image']
    image_spacing = image_dict['spacing']
    mask = mask_dict['npy_image']

    # coarse_size = [170, 170, 170]
    # coarse_image, _ = ScipyResample.resample_to_size(image, coarse_size)
    # coarse_mask, _ = ScipyResample.resample_mask_to_size(mask, coarse_size,
    #                                                      num_label=13)
    # data_augment_test(coarse_image, coarse_mask, [160, 160, 160])

    margin = [int(40 / image_spacing[0]),
              int(40 / image_spacing[1]),
              int(40 / image_spacing[2])]
    crop_image, crop_mask = crop_image_according_to_mask(image, mask, margin)
    fine_size = [176, 212, 212]
    fine_image, _ = ScipyResample.resample_to_size(crop_image, fine_size)
    fine_mask, _ = ScipyResample.resample_mask_to_size(crop_mask, fine_size,
                                                       num_label=13)
    data_augment_test(fine_image, fine_mask, [160, 192, 192])


