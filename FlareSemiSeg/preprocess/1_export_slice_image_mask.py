import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback
from multiprocessing import Pool

from Common.transforms.image_io import load_sitk_image
from Common.fileio.file_utils import read_txt


image_dir = "/data/zhangfan/FLARE2022/raw_data/crop_unlabeled_image/"
mask_dir = "/data/zhangfan/FLARE2022/pseudo_mask/iters0_mask/"
out_dir = "/data/zhangfan/FLARE2022/pseudo_mask/vis_iters0_mask/"
file_path = "/data/zhangfan/FLARE2022/pseudo_mask/iters0_reliable_score/reliable_series.txt"


def export_image(file):
    uid = file.split('.nii.gz')[0]
    image_path = image_dir + uid + '.nii.gz'
    mask_path = mask_dir + file
    out_path = out_dir + uid + '.jpg'
    image_dict = load_sitk_image(image_path)
    mask_dict = load_sitk_image(mask_path)
    image = image_dict['npy_image']
    mask = mask_dict['npy_image']

    zz, yy, xx = np.where(mask != 0)
    x_mid_slice = int(np.median(xx))

    image_slice = np.flip(image[:, :, x_mid_slice], axis=0)
    mask_slice = np.flip(mask[:, :, x_mid_slice], axis=0)
    masked = np.ma.masked_where(mask_slice == 0, mask_slice)

    plt.figure()
    plt.imshow(image_slice, cmap='gray')
    plt.imshow(masked, cmap='jet', alpha=0.5)
    # plt.show()
    plt.savefig(out_path)


if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# file_names = os.listdir(mask_dir)
file_names = read_txt(file_path)
pool = Pool(20)
for data in file_names:
    data += '.nii.gz'
    if not data.endswith('.nii.gz'):
        continue
    try:
        pool.apply_async(export_image, (data,))
    except Exception as err:
        traceback.print_exc()

pool.close()
pool.join()
