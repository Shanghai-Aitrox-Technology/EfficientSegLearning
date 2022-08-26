
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import scipy.ndimage.morphology as morphology

from Common.transforms.mask_process import keep_single_channel_cc_mask
from Common.transforms.image_io import load_sitk_image, save_sitk_from_npy


def smooth_mask(mask_path, out_path, morp_iters=2):
    mask_dict = load_sitk_image(mask_path)
    mask = mask_dict['npy_image']

    out_mask = np.zeros(mask.shape, np.uint8)
    for i in range(1, 14):
        temp_mask = mask.copy()
        temp_mask = np.where(temp_mask == i, 1, 0)
        struct = morphology.generate_binary_structure(3, 1)
        temp_mask = morphology.binary_closing(temp_mask, structure=struct, iterations=morp_iters)
        temp_mask = temp_mask.astype(np.uint8)
        keep_single_channel_cc_mask(temp_mask, 1, 500, out_mask, i)

    save_sitk_from_npy(out_mask, out_path, spacing=mask_dict['spacing'][::-1], use_compression=True)
    print(f'process {mask_path} finish!')


if __name__ == '__main__':
    mask_dir = "/data/zhangfan/FLARE2022/raw_data/pseudo_mask/"
    out_dir = "/data/zhangfan/FLARE2022/raw_data/smooth_mask/"

    file_names = os.listdir(mask_dir)
    pool = Pool(int(cpu_count() * 0.7))
    for file_name in file_names:
        mask_path = mask_dir + file_name
        out_path = out_dir + file_name
        try:
            pool.apply_async(smooth_mask, (mask_path, out_path))
        except Exception as err:
            print(err)

    pool.close()
    pool.join()
