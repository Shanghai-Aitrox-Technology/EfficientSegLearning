import os
import sys
import json
import traceback
import numpy as np
from multiprocessing import Pool

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from Common.transforms.image_io import load_sitk_image
from Common.fileio.file_utils import read_txt, write_csv
from BaseSeg.evaluation.surface_dice import compute_surface_distances, \
    compute_dice_coefficient, compute_robust_hausdorff


num_class = 13
object_labels = ['liver', 'right kidney', 'spleen', 'pancreas', 'aorta', 'inferior vena cava',
                 'right adrenal gland', 'left adrenal gland', 'gallbladder', 'esophagus',
                 'stomach', 'duodenum', 'left kidney']


def process_compute_metric(name, gt_dir, predict_dir, csv_path):
    print('process {} start...'.format(name))
    gt_mask_path = gt_dir + name + '.nii.gz'
    predict_mask_path = predict_dir + name + '.nii.gz'
    if not os.path.exists(gt_mask_path) or not os.path.exists(predict_mask_path):
        return

    gt_dict = load_sitk_image(gt_mask_path)
    predict_dict = load_sitk_image(predict_mask_path)
    gt_mask = gt_dict['npy_image'].astype(np.uint8)
    predict_mask = predict_dict['npy_image'].astype(np.uint8)
    spacing = gt_dict['spacing']

    temp_mask = gt_mask.copy()
    end_idx = np.max(temp_mask)
    temp_mask[temp_mask == 0] = 30
    start_idx = np.min(temp_mask)

    predict_mask = predict_mask.astype(np.uint8)
    gt_mask = gt_mask.astype(np.uint8)

    area_dice = np.array([-1.0]*num_class)
    surface_dice = np.array([-1.0]*num_class)
    for i in range(start_idx, end_idx+1):
        gt_mask_i = np.zeros_like(gt_mask)
        gt_mask_i[gt_mask == i] = 1
        predict_mask_i = np.zeros_like(predict_mask)
        predict_mask_i[predict_mask == i] = 1

        surface_distances = compute_surface_distances(gt_mask_i, predict_mask_i, spacing)
        dice = compute_dice_coefficient(gt_mask_i, predict_mask_i)
        hausdorff = compute_robust_hausdorff(surface_distances, 95)
        area_dice[i-1] = dice
        surface_dice[i-1] = hausdorff

    out_content = [name, spacing[0]]
    total_area_dice = 0
    total_surface_dice = 0
    num_target = 0
    for i in range(num_class):
        out_content.append(area_dice[i])
        out_content.append(surface_dice[i])
        if area_dice[i] != -1:
            total_area_dice += area_dice[i]
            total_surface_dice += surface_dice[i]
            num_target += 1
            print('{} DSC: {}, NSC: {}'.format(object_labels[i], area_dice[i], surface_dice[i]))
    out_content.extend([total_area_dice / num_target, total_surface_dice / num_target])
    write_csv(csv_path, out_content, mul=False, mod='a+')
    print('Average_DSC: {}, Average_NSC: {}'.format(total_area_dice / num_target, total_surface_dice / num_target))
    print('process {} finish!'.format(name))


if __name__ == '__main__':
    series_uid_path = '/data/zhangfan/FLARE2022/train_data/full_crop_train/file_list/val_series_uid.txt'
    gt_mask_dir = '/data/zhangfan/FLARE2022/raw_data/labeled_mask/'
    predict_mask_dir = '/data/zhangfan/FLARE2022/results/val_mask/'
    out_ind_csv_dir = '/data/zhangfan/FLARE2022/results/val_mask/'
    out_ind_csv_path = out_ind_csv_dir + 'base_flare_val_metric.csv'

    ind_content = ['series_uid', 'z_spacing']
    object_metric = []
    for object_name in object_labels:
        object_metric.extend([object_name + '_DSC', object_name + '_NSC'])
    ind_content.extend(object_metric)
    ind_content.extend(['Average_DSC', 'Average_NSC'])
    write_csv(out_ind_csv_path, ind_content, mul=False, mod='w')

    file_names = read_txt(series_uid_path)
    pool = Pool(10)
    for file_name in file_names:
        try:
            pool.apply_async(process_compute_metric, (file_name, gt_mask_dir,
                                                      predict_mask_dir, out_ind_csv_path))
        except Exception as err:
            traceback.print_exc()
    pool.close()
    pool.join()