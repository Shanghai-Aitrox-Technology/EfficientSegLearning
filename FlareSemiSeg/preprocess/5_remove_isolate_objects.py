
import os
import numpy as np
from skimage import measure
from skimage.morphology import label

from Common.transforms.image_io import load_sitk_image


if __name__ == '__main__':
    image_dir = '/data/zhangfan/FLARE2022/results/test_mask_base/'
    file_names = os.listdir(image_dir)

    badcase = ['FLARETs_0002_0000']
    for file_name in file_names:
        if file_name.endswith('_coarse.nii.gz') and file_name.split('_coarse.nii.gz')[0] in badcase:
            image_path = image_dir + file_name
            image_dict = load_sitk_image(image_path)
            mask = image_dict['npy_image']
            labeled_mask = mask.copy()
            region_props = measure.regionprops(labeled_mask)
            centroids = {}
            bboxs = []
            for i in range(len(region_props)):
                centroids[i + 1] = region_props[i].centroid[0]
                bboxs.append(region_props[i].bbox)

            sorted_centroids = sorted(centroids.items(), key=lambda item: item[1], reverse=False)
            sorted_bboxs = []
            bboxs = np.array(bboxs)
            for i in range(6):
                sorted_bbox = sorted(bboxs[:, i], reverse=False)
                sorted_bboxs.append(sorted_bbox)

            print('debug')