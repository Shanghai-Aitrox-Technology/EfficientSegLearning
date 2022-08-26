
import os
import sys
import numpy as np
import SimpleITK as sitk

import traceback
from multiprocessing import Pool, cpu_count

import cc3d
import fastremap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from Common.transforms.image_reorient import reorient_image_to_RAS
from Common.transforms.image_io import load_sitk_image
from Common.transforms.mask_process import extract_bbox


def remove_background(image):
    mask = image.copy()
    mask = np.where(mask > -500, 1, 0).astype(np.uint8)
    mask_shape = mask.shape
    centroid = [mask_shape[0] // 2, mask_shape[1] // 2, mask_shape[2] // 2]
    mask = cc3d.connected_components(mask, connectivity=6)
    label = mask[centroid[0], centroid[1], centroid[2]]
    if label != 0:
        mask[mask != label] = 0
    else:
        areas = {}
        for label, extracted in cc3d.each(mask, binary=True, in_place=True):
            areas[label] = fastremap.foreground(extracted)

        candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)
        mask = np.where(mask == int(candidates[0][0]), 1, 0)

    bbox = extract_bbox(mask)
    crop_image = image[bbox[0, 0]:bbox[0, 1],
                       bbox[1, 0]:bbox[1, 1],
                       bbox[2, 0]:bbox[2, 1]]

    return crop_image, bbox


def reorient_threshold_seg(image_path, out_image, mask_path=None, out_mask=None):
    print(f'start process: {image_path}')

    image_dict = load_sitk_image(image_path)
    sitk_image = image_dict['sitk_image']
    sitk_image = reorient_image_to_RAS(sitk_image, interpolate_mode='linear')
    image = sitk.GetArrayFromImage(sitk_image)

    crop_image, bbox = remove_background(image)
    out_sitk_image = sitk.GetImageFromArray(crop_image)
    out_sitk_image.SetSpacing(sitk_image.GetSpacing())
    sitk.WriteImage(out_sitk_image, out_image, True)

    if mask_path is not None:
        mask_dict = load_sitk_image(mask_path)
        sitk_mask = mask_dict['sitk_image']
        sitk_mask = reorient_image_to_RAS(sitk_mask, interpolate_mode='nearest')
        mask = sitk.GetArrayFromImage(sitk_mask)
        crop_mask = mask[bbox[0, 0]:bbox[0, 1],
                         bbox[1, 0]:bbox[1, 1],
                         bbox[2, 0]:bbox[2, 1]]
        crop_mask = crop_mask.astype(np.uint8)
        out_sitk_mask = sitk.GetImageFromArray(crop_mask)
        out_sitk_mask.SetSpacing(sitk_mask.GetSpacing())
        sitk.WriteImage(out_sitk_mask, out_mask, True)

    print(f'end process: {out_image}')


if __name__ == '__main__':
    # src_image_dir = '../raw_data/labeled_image/'
    # src_mask_dir = '../raw_data/labeled_mask/'
    # out_image_dir = '../raw_data/crop_labeled_image/'
    # out_mask_dir = '../raw_data/crop_labeled_mask/'
    src_image_dir = '../raw_data/unlabeled_image/'
    src_mask_dir = '../raw_data/unlabeled_mask/'
    out_image_dir = '../raw_data/crop_unlabeled_image/'
    out_mask_dir = '../raw_data/crop_unlabeled_mask/'

    if not os.path.exists(out_image_dir):
        os.makedirs(out_image_dir)
    if not os.path.exists(out_mask_dir):
        os.makedirs(out_mask_dir)

    file_names = os.listdir(src_image_dir)
    pool = Pool(int(cpu_count()*0.5))
    for data in file_names:
        src_image_path = src_image_dir + data
        dst_image_path = out_image_dir + data
        src_mask_path = src_mask_dir + data
        if not os.path.exists(src_mask_path):
            src_mask_path = None
            dst_mask_path = None
        else:
            dst_mask_path = out_mask_dir + data
        try:
            pool.apply_async(reorient_threshold_seg, (src_image_path, dst_image_path, src_mask_path, dst_mask_path))
        except Exception as err:
            traceback.print_exc()

    pool.close()
    pool.join()