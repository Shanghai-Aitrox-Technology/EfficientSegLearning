
import os
import json
import numpy as np
import SimpleITK as sitk
from typing import Dict
from multiprocessing import Pool, cpu_count


def get_target_bbox(mask: np.ndarray) -> np.ndarray:
    """
    Get the bounding box from target mask.
    :param mask:
    :return:
    """
    zz, yy, xx = np.where(mask > 0)

    bbox = np.array([[np.min(zz), np.max(zz)],
                     [np.min(yy), np.max(yy)],
                     [np.min(xx), np.max(xx)]])
    return bbox


def export_data_features(series_uid, image_path, mask_path, out_path) -> Dict:
    """
    Export features from source image or mask.
    :param series_uid: series uid.
    :param image_path: path of source image.
    :param mask_path: path of source mask.
    :param out_path: path of out features, json format.
    :return:
    res_dict:{'uid', 'spacing', 'direction', 'index_size', 'physical_size', 'label_range',
              'label_volume', 'label_intensity', 'binary_scale', 'multi_label_scale'}
    """
    print(f'start process: {series_uid}')
    res_dict = {}

    sitk_image = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(sitk_image)
    spacing = list(reversed(sitk_image.GetSpacing()))
    direction = list(sitk_image.GetDirection())
    diag_direction = [direction[0], direction[4], direction[8]]
    image_index_size = image.shape
    image_physical_size = [spacing[i] * image_index_size[i] for i in range(3)]

    res_dict['uid'] = series_uid
    res_dict['spacing'] = spacing
    res_dict['direction'] = diag_direction
    res_dict['index_size'] = image_index_size
    res_dict['physical_size'] = image_physical_size

    if os.path.exists(mask_path):
        sitk_mask = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(sitk_mask)
        temp_mask = mask.copy()
        end_idx = np.max(temp_mask)
        temp_mask[temp_mask == 0] = 10000
        start_idx = np.min(temp_mask)
        res_dict['label_range'] = [float(start_idx), float(end_idx)]

        mask_volume = [-10000.]*(end_idx-start_idx+1)
        mask_intensity = [-10000.]*(end_idx-start_idx+1)
        for i in range(start_idx, end_idx + 1):
            num = len(np.where(mask == i)[0])
            if num == 0:
                continue
            volume = float(num * spacing[0] * spacing[1] * spacing[2])
            mask_volume[i-1] = volume
            fg = image[mask == i]
            mask_intensity[i-1] = float(np.mean(fg))
        res_dict['label_volume'] = mask_volume
        res_dict['label_intensity'] = mask_intensity

        binary_mask = mask.copy()
        binary_mask[binary_mask != 0] = 1
        binary_bbox = get_target_bbox(binary_mask)
        binary_scale = [spacing[0] * (binary_bbox[0, 1] - binary_bbox[0, 0] + 1),
                        spacing[1] * (binary_bbox[1, 1] - binary_bbox[1, 0] + 1),
                        spacing[2] * (binary_bbox[2, 1] - binary_bbox[2, 0] + 1)]
        res_dict['binary_scale'] = binary_scale

        multi_label_scale = [[] for _ in range(end_idx-start_idx+1)]
        for i in range(start_idx, end_idx + 1):
            temp_mask = mask.copy()
            temp_mask = np.where(temp_mask == i, 1, 0)
            if np.sum(temp_mask) == 0:
                continue
            bbox = get_target_bbox(temp_mask)
            scale = [spacing[0] * (bbox[0, 1] - bbox[0, 0] + 1),
                     spacing[1] * (bbox[1, 1] - bbox[1, 0] + 1),
                     spacing[2] * (bbox[2, 1] - bbox[2, 0] + 1)]
            multi_label_scale[i-1] = scale
        res_dict['multi_label_scale'] = multi_label_scale

    with open(out_path, "w") as f:
        json.dump(res_dict, f)

    print(f'end process: {series_uid}')

    return res_dict


if __name__ == '__main__':
    image_dir = "/data/zhangfan/FLARE2022/raw_data/labeled_image/"
    mask_dir = "/data/zhangfan/FLARE2022/raw_data/labeled_mask/"
    out_dir = "/data/zhangfan/FLARE2022/preprocess_data/feature/"
    file_names = os.listdir(image_dir)

    pool = Pool(int(cpu_count() * 0.7))
    for file_name in file_names:
        uid = file_name.split('.nii.gz')[0]
        image_path = image_dir + file_name
        mask_path = mask_dir + file_name
        out_path = out_dir + uid + '.json'
        # if os.path.exists(out_path):
        #     continue
        try:
            pool.apply_async(export_data_features, (uid, image_path, mask_path, out_path))
        except Exception as err:
            print(err)

    pool.close()
    pool.join()
