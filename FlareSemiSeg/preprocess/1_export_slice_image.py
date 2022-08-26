import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback
from multiprocessing import Pool

from Common.transforms.image_io import load_sitk_image


image_dir = "/data/zhangfan/FLARE2022/raw_data/crop_labeled_image/"
out_dir = "/data/zhangfan/FLARE2022/preprocess_data/vis_crop_labeled_fig/"


def export_image(file):
    uid = file.split('.nii.gz')[0]
    image_path = image_dir + uid + '.nii.gz'
    out_path = out_dir + uid + '.jpg'
    image_dict = load_sitk_image(image_path)
    image = image_dict['npy_image']
    image_shape = image.shape[2]
    image_slice = np.flip(image[:, :, image_shape//2], axis=0)

    plt.figure()
    plt.imshow(image_slice, cmap='gray')
    # plt.show()
    plt.savefig(out_path)


if not os.path.exists(out_dir):
    os.mkdir(out_dir)

file_names = os.listdir(image_dir)
pool = Pool(20)
for data in file_names:
    if not data.endswith('.nii.gz'):
        continue
    try:
        pool.apply_async(export_image, (data,))
    except Exception as err:
        traceback.print_exc()

pool.close()
pool.join()
