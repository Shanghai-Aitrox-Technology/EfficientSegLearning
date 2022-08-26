import os
import warnings
import datetime

import cc3d
import fastremap

import numpy as np
from skimage import measure
from skimage.morphology import label
from multiprocessing import Process

from .dataset import FlareInferDataset
from BaseSeg.engine.seg_model import SegInferModelV2

from Common.utils.logger import get_logger
from Common.fileio.file_utils import read_txt, write_csv
from Common.transforms.image_io import save_sitk_from_npy
from Common.utils.profiling import get_gpu_memory_usage, get_time
from Common.transforms.mask_process import extract_bbox, crop_image_according_to_bbox,\
    mapping_mask_to_raw_roi


class FlareInfer(object):
    def __init__(self, cfg):
        super(FlareInfer, self).__init__()

        # step 1 >>> init params
        self.cfg = cfg
        self.save_dir = self.cfg.testing.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # step 2 >>> init model
        self.coarse_model = SegInferModelV2(self.cfg.coarse_model, 'flare_coarse_seg')
        self.fine_model = SegInferModelV2(self.cfg.fine_model, 'flare_fine_seg')
        if self.cfg.testing.multi_stage == 3:
            self.part_model = SegInferModelV2(self.cfg.part_model, 'flare_part_seg')

        # step 3 >>> init log
        self.log_dir = self.save_dir + '/logs_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.logger = get_logger(self.log_dir)
        self.logger.info('\n------------ inference options -------------')
        self.logger.info(str(self.cfg.pretty_text))
        self.logger.info('-------------- End ---------------------------\n')

        # step 4 >>> init monitor gpu memory usage
        self.gpu_memory_txt_path = os.path.join(self.log_dir, 'max_gpu_memory_usage.txt')
        p = Process(target=get_gpu_memory_usage,
                    args=(0.1, True,
                          self.cfg.env.rank,
                          self.gpu_memory_txt_path))
        p.daemon = True
        p.start()

        # step 5 >>> init timer
        self.all_time_consume = {'pipeline': [], 'coarse_pre': [], 'coarse_infer': [], 'coarse_post': [],
                                 'fine_infer': [], 'fine_post': [], 'data_load': []}
        self.time_txt_path = os.path.join(self.log_dir, 'time_consume.csv')
        write_csv(self.time_txt_path, ['series_uid', 'time_consume'], mul=False, mod='w')

    def _update_timer(self, item, time_consume):
        if item in self.all_time_consume.keys():
            self.all_time_consume[item].append(time_consume)
            self.logger.info(f'{item} time: {time_consume} s')
        else:
            warnings.warn(f'{item} not in all_time_dict, update timer failed!')

    def _get_target_bbox(self, mask, is_labeled=True):
        labeled_mask = mask.copy()
        if not is_labeled:
            labeled_mask = label(labeled_mask, neighbors=8, background=0, return_num=False)
        region_props = measure.regionprops(labeled_mask)

        bbox = []
        for i in range(len(region_props)):
            bbox.append(region_props[i].bbox)

        bbox = np.array(bbox)
        sorted_bbox = []
        for i in range(6):
            sorted_bbox.append(sorted(bbox[:, i], reverse=False))

        if len(region_props) > 5:
            out_bbox = [sorted_bbox[0][1], sorted_bbox[3][-2],
                        sorted_bbox[1][0], sorted_bbox[4][-1],
                        sorted_bbox[2][0], sorted_bbox[5][-1]]
        else:
            out_bbox = [sorted_bbox[0][0], sorted_bbox[3][-1],
                        sorted_bbox[1][0], sorted_bbox[4][-1],
                        sorted_bbox[2][0], sorted_bbox[5][-1]]

        return out_bbox

    def _remove_background(self, image):
        mask = np.where(image > -500, 1, 0).astype(np.uint8)
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

    def run(self):
        test_dataset = FlareInferDataset(self.cfg)
        self.logger.info('Starting test...')
        self.logger.info('test samples: {}'.format(len(test_dataset)))

        t_iter_start = get_time()
        for idx, data_dict in enumerate(test_dataset):
            data_dict = data_dict[0] if isinstance(data_dict, list) else data_dict
            series_id = data_dict['series_id']
            self.logger.info(f'process: {idx}/{len(test_dataset)}, \nseries_uid:{series_id}')
            raw_image = data_dict['image']
            sitk_image = data_dict['sitk_image']
            raw_spacing = data_dict['raw_spacing']
            raw_image_shape = raw_image.shape
            t_data_load = get_time()
            time_consume = t_data_load - t_iter_start
            self._update_timer('data_load', time_consume)

            # step 0 >>> remove background.
            if self.cfg.testing.remove_background:
                crop_bg_image, crop_bg_bbox = self._remove_background(raw_image)
                crop_bg_size = crop_bg_image.shape
                coarse_zoom_factor = crop_bg_size / np.array(self.cfg.coarse_model.input_size)
            else:
                crop_bg_image = raw_image
                crop_bg_size = crop_bg_image.shape
                coarse_zoom_factor = crop_bg_size / np.array(self.cfg.coarse_model.input_size)
                crop_bg_bbox = np.zeros([3, 2], np.uint8)

            # step 1 >>> segmentation in rough location in low resolution.
            self.logger.info('coarse segmentation start...')
            coarse_image, time_consume = self.coarse_model.preprocess(crop_bg_image)
            self._update_timer('coarse_pre', time_consume)

            coarse_image, time_consume = self.coarse_model.infer(coarse_image)
            self._update_timer('coarse_infer', time_consume)

            coarse_spacing = [raw_spacing[i] * coarse_zoom_factor[i] for i in range(3)]
            coarse_image, time_consume = self.coarse_model.postprocess(coarse_image, coarse_spacing)
            self._update_timer('coarse_post', time_consume)

            # crop image based rough mask.
            is_labeled = False if self.cfg.coarse_model.num_class == 1 else True
            rough_bbox = self._get_target_bbox(coarse_image, is_labeled)
            rough_bbox = [int(rough_bbox[0] * coarse_zoom_factor[0]), int(rough_bbox[1] * coarse_zoom_factor[0]),
                          int(rough_bbox[2] * coarse_zoom_factor[1]), int(rough_bbox[3] * coarse_zoom_factor[1]),
                          int(rough_bbox[4] * coarse_zoom_factor[2]), int(rough_bbox[5] * coarse_zoom_factor[2])]
            margin = [self.cfg.data_loader.margin / raw_spacing[i] for i in range(3)]
            crop_image, crop_fine_bbox = crop_image_according_to_bbox(crop_bg_image, rough_bbox, margin)
            crop_fine_image = crop_image.copy()
            self.logger.info('coarse segmentation complete!')

            # step 2 >>> fine segmentation according to rough spine bbox.
            raw_fine_shape = crop_image.shape
            fine_zoom_factor = raw_fine_shape / np.array(self.cfg.fine_model.input_size)
            fine_spacing = [raw_spacing[i] * fine_zoom_factor[i] for i in range(3)]

            self.logger.info('fine segmentation start...')
            crop_image, time_consume = self.fine_model.preprocess(crop_image)
            crop_image, time_consume = self.fine_model.infer(crop_image)
            self._update_timer('fine_infer', time_consume)

            output_spacing = fine_spacing if not self.cfg.fine_model.ahead_resample else raw_spacing
            crop_image, time_consume = self.fine_model.postprocess(crop_image, output_spacing)
            self._update_timer('fine_post', time_consume)

            if self.cfg.testing.multi_stage == 3:
                crop_image, part_time = self._run_part(crop_fine_image, crop_image, margin, raw_spacing)
                self.logger.info(f'part seg time: {part_time}')

            crop_mapping_bbox = [crop_bg_bbox[0, 0]+crop_fine_bbox[0],
                                 crop_bg_bbox[0, 0]+crop_fine_bbox[1],
                                 crop_bg_bbox[1, 0]+crop_fine_bbox[2],
                                 crop_bg_bbox[1, 0]+crop_fine_bbox[3],
                                 crop_bg_bbox[2, 0]+crop_fine_bbox[4],
                                 crop_bg_bbox[2, 0]+crop_fine_bbox[5]]
            out_mask = mapping_mask_to_raw_roi(crop_image, crop_mapping_bbox, raw_image_shape)
            self.logger.info('fine segmentation complete!')

            # step 3 >>> save seg mask.
            mask_path = os.path.join(self.save_dir, series_id + ".nii.gz")
            save_sitk_from_npy(out_mask, mask_path, spacing=sitk_image.GetSpacing(),
                               origin=sitk_image.GetOrigin(),
                               direction=sitk_image.GetDirection(),
                               use_compression=self.cfg.testing.is_compress_mask)
            if self.cfg.testing.is_save_coarse_mask:
                mask_path = os.path.join(self.save_dir, series_id + "_coarse.nii.gz")
                save_sitk_from_npy(coarse_image, mask_path,
                                   use_compression=self.cfg.testing.is_compress_mask)

            self.logger.info(f'Process {series_id} complete!')

            t_iter_end = get_time()
            time_consume = t_iter_end - t_iter_start
            self._update_timer('pipeline', time_consume)
            t_iter_start = t_iter_end

            write_csv(self.time_txt_path, [series_id, time_consume], mul=False, mod='a+')

        gpu_info = read_txt(self.gpu_memory_txt_path)
        max_memory_usage = float(gpu_info[0])
        self.logger.info("Max gpu-memory usage: {} MB".format(max_memory_usage))
        for key, value in self.all_time_consume.items():
            if value is None:
                continue
            self.logger.info(f'{key} average time: {np.mean(np.array(value))} s')

    def _run_part(self, image, mask, margin, spacing):
        self.part_label = [7, 8, 9, 10, 12]
        part_mask = mask.copy()
        for label in range(1, 14):
            if label not in self.part_label:
                part_mask[part_mask == label] = 0

        temp_bbox = extract_bbox(part_mask)
        raw_bbox = [int(temp_bbox[0][0]), int(temp_bbox[0][1]),
                    int(temp_bbox[1][0]), int(temp_bbox[1][1]),
                    int(temp_bbox[2][0]), int(temp_bbox[2][1])]
        crop_image, crop_bbox = crop_image_according_to_bbox(image, raw_bbox, margin)
        raw_image_shape = image.shape
        predict_mask, time_consume = self.part_model.run(crop_image, spacing)
        predict_mask = mapping_mask_to_raw_roi(predict_mask, crop_bbox, raw_image_shape)

        for idx, label in enumerate(self.part_label):
            mask[mask == label] = 0
            mask[predict_mask == idx+1] = label

        return mask, time_consume