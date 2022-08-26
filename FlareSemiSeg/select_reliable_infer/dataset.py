
import os
import numpy as np
import SimpleITK as sitk

from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator

from Common.fileio.file_utils import read_txt, split_filename
from Common.transforms.image_io import load_sitk_image
from Common.transforms.image_resample import ScipyResample
from Common.transforms.image_transform import clip_and_normalize_mean_std
from Common.transforms.image_reorient import reorient_image_to_RAS


def test_collate_fn(batch):
    """
    Collate function in test phase.
    """
    return batch


class DataLoaderX(DataLoader):
    """
    Prefetch generator dataset.
    """
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def build_dataset(cfg):
    return FlareInferDataset(cfg)


class FlareInferDataset(Dataset):
    """
        Dataset of flare segmentation in prediction phase.
    """
    def __init__(self, cfg):
        super(FlareInferDataset, self).__init__()
        self.cfg = cfg
        self.image_dir = cfg.data_loader.test_image_dir
        self.data_filenames = os.listdir(self.image_dir)
        self.window_level = cfg.data_loader.window_level
        test_uid_txt = self.cfg.data_loader.test_series_uid_txt
        if test_uid_txt is not None and os.path.exists(test_uid_txt):
            all_series_uid = read_txt(test_uid_txt)
            filenames = []
            for file_name in self.data_filenames:
                series_id = split_filename(file_name)[0]
                if series_id in all_series_uid:
                    filenames.append(file_name)
            self.data_filenames = filenames

    def __len__(self):
        return len(self.data_filenames)

    def __getitem__(self, idx: int) -> dict:
        file_name = self.data_filenames[idx]
        image_path = self.image_dir + file_name
        series_id = split_filename(file_name)[0]

        image_dict = load_sitk_image(image_path)
        sitk_image = image_dict['sitk_image']
        raw_image = image_dict['npy_image']
        raw_spacing = image_dict['spacing']
        image_direction = image_dict['direction']

        if self.cfg.data_loader.is_norma_direction:
            sitk_image = reorient_image_to_RAS(sitk_image)
            raw_image = sitk.GetArrayFromImage(sitk_image)
            raw_spacing = list(reversed(sitk_image.GetSpacing()))
            image_direction = list(sitk_image.GetDirection())

        return {'series_id': series_id,
                'image': raw_image,
                'sitk_image': sitk_image,
                'raw_spacing': raw_spacing,
                'direction': image_direction}