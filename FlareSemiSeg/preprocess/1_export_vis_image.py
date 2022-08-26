
import os
import numpy as np
from tqdm import tqdm

image_dir = '/data/zhangfan/FLARE2022/train_data_part/full_crop_train/part_image/176_222_222/'
mask_dir = '/data/zhangfan/FLARE2022/train_data_part/full_crop_train/part_mask/176_222_222/'
out_image_dir = '/data/zhangfan/FLARE2022/train_data_part/full_crop_train/vis_image/'
out_mask_dir = '/data/zhangfan/FLARE2022/train_data_part/full_crop_train/vis_mask/'


def save_image(data, path):
    import SimpleITK as sitk

    sitk_image = sitk.GetImageFromArray(data)
    sitk.WriteImage(sitk_image, path, True)

if not os.path.exists(out_image_dir):
    os.makedirs(out_image_dir)
if not os.path.exists(out_mask_dir):
    os.makedirs(out_mask_dir)

file_names = os.listdir(image_dir)
for file_name in tqdm(file_names):
    src_image_path = image_dir + file_name
    src_mask_path = mask_dir + file_name
    src_image = np.load(src_image_path)
    src_mask = np.load(src_mask_path)
    dst_image_path = out_image_dir + file_name.split('.npy')[0] + '.mha'
    dst_mask_path = out_mask_dir + file_name.split('.npy')[0] + '.mha'

    save_image(src_image, dst_image_path)
    save_image(src_mask, dst_mask_path)