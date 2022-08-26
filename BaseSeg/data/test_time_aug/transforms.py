
import torch
import numpy as np
from typing import Optional, List, Union
from . import functional as F
from .base import DualTransform, ImageOnlyTransform


class Flip(DualTransform):
    """Flip images"""
    identity_param = -1

    def __init__(self, axis: List[int]):
        if self.identity_param not in axis:
            axis = [self.identity_param] + axis
        super().__init__("flip_axis", axis)

    def apply_aug_image(self, image, flip_axis=-1, **kwargs):
        if flip_axis != self.identity_param:
            image = F.flip(image, axis=flip_axis)

        return image

    def apply_deaug_mask(self, mask, flip_axis=-1, **kwargs):
        if isinstance(mask, tuple) and len(mask) >= 2:
            mask = mask[0]
        if flip_axis != self.identity_param:
            mask = F.flip(mask, axis=flip_axis)

        mask_shape = mask.shape
        target_bbox = [[0, mask_shape[2]], [0, mask_shape[3]], [0, mask_shape[4]]]

        return mask, target_bbox


class Rotate90(DualTransform):
    """Rotate images"""
    identity_param = 0

    def __init__(self, angles: List[int], axis=(2, 3)):
        super().__init__()
        if self.identity_param not in angles:
            angles = [self.identity_param] + angles
        self.axis = axis

        super().__init__("angle", angles)

    def apply_aug_image(self, image, angle=0, **kwargs):
        k = angle // 90 if angle >= 0 else (angle + 360) // 90
        if k != self.identity_param:
            image = F.rot90(image, k, self.axis)

        return image

    def apply_deaug_mask(self, mask, angle=0, **kwargs):
        if isinstance(mask, tuple) and len(mask) >= 2:
            mask = mask[0]

        mask = self.apply_aug_image(mask, -angle)

        mask_shape = mask.shape
        target_bbox = [[0, mask_shape[2]], [0, mask_shape[3]], [0, mask_shape[4]]]

        return mask, target_bbox


class Crop(DualTransform):
    """Crop image"""
    identity_param = 'crop_zzz'

    def __init__(self, crop_mode: List[str], crop_ratio=0.1):
        super().__init__()
        if self.identity_param not in crop_mode:
            crop_mode = [self.identity_param] + crop_mode
        self.crop_ratio = crop_ratio

        super().__init__("mode", crop_mode)

    def apply_aug_image(self, image, mode='crop_ccc', **kwargs):
        assert mode in ['crop_sss', 'crop_eee', 'crop_ess', 'crop_ses', 'crop_sse',
                        'crop_see', 'crop_ese', 'crop_ees', 'crop_ccc', 'crop_zzz']
        h, w, d = image.shape[2:]
        crop_h, crop_w, crop_d = h-int(h*self.crop_ratio), w-int(w*self.crop_ratio), d-int(d*self.crop_ratio)
        crop_bbox = F.crop_registry.get(mode)(image, crop_h, crop_w, crop_d)
        self.crop_bbox = crop_bbox
        self.image_shape = image.shape[2:]

        return image[:, :, crop_bbox[0][0]:crop_bbox[0][1],
                     crop_bbox[1][0]:crop_bbox[1][1],
                     crop_bbox[2][0]:crop_bbox[2][1]]

    def apply_deaug_mask(self, mask, mode='crop_ccc', **kwargs):
        if isinstance(mask, tuple) and len(mask) >= 2:
            mask = mask[0]

        crop_bbox = self.crop_bbox

        return mask, crop_bbox


class Scale(DualTransform):
    """Scale images"""
    identity_param = 1

    def __init__(
        self,
        scales: List[Union[int, float]],
        interpolation: str = "trilinear",
        align_corners: Optional[bool] = True,
    ):
        if self.identity_param not in scales:
            scales = [self.identity_param] + scales
        self.interpolation = interpolation
        self.align_corners = align_corners

        super().__init__("scale", scales)

    def apply_aug_image(self, image, scale=1, **kwargs):
        self.raw_size = list(image.shape[2:])
        if scale != self.identity_param:
            self.scale_size = [int(scale * item) for item in self.raw_size]
            image = F.resize(image, self.scale_size,
                             interpolation=self.interpolation,
                             align_corners=self.align_corners)
            image, target_bbox, crop_or_pad = F.crop_pad(image, self.raw_size)
            self.target_bbox = target_bbox
            self.crop_or_pad = crop_or_pad

        return image

    def apply_deaug_mask(self, mask, scale=1, **kwargs):
        if isinstance(mask, tuple) and len(mask) >= 2:
            mask = mask[0]

        if scale != self.identity_param:
            if self.crop_or_pad == 2:
                mask = mask[:, :, self.target_bbox[0][0]:self.target_bbox[0][1],
                            self.target_bbox[1][0]:self.target_bbox[1][1],
                            self.target_bbox[2][0]:self.target_bbox[2][1]]
                out_bbox = [[0, self.raw_size[0]], [0, self.raw_size[1]], [0, self.raw_size[2]]]
                out_size = self.raw_size
            else:
                out_bbox = [[int(self.target_bbox[0][0]*1.0/scale), int(self.target_bbox[0][1]*1.0/scale)],
                            [int(self.target_bbox[1][0]*1.0/scale), int(self.target_bbox[1][1]*1.0/scale)],
                            [int(self.target_bbox[2][0]*1.0/scale), int(self.target_bbox[2][1]*1.0/scale)]]
                out_size = [out_bbox[0][1]-out_bbox[0][0],
                            out_bbox[1][1]-out_bbox[1][0],
                            out_bbox[2][1]-out_bbox[2][0]]
            mask = F.resize(mask, out_size,
                            interpolation=self.interpolation,
                            align_corners=self.align_corners)
        else:
            out_bbox = [[0, self.raw_size[0]], [0, self.raw_size[1]], [0, self.raw_size[2]]]

        return mask, out_bbox


class Add(ImageOnlyTransform):
    """Add value to images

    Args:
        values (List[float]): values to add to each pixel
    """

    identity_param = 0

    def __init__(self, values: List[float]):

        if self.identity_param not in values:
            values = [self.identity_param] + list(values)
        super().__init__("value", values)

    def apply_aug_image(self, image, value=0, **kwargs):
        if value != self.identity_param:
            image = F.add(image, value)
        return image


class Multiply(ImageOnlyTransform):
    """Multiply images by factor

    Args:
        factors (List[float]): factor to multiply each pixel by
    """

    identity_param = 1

    def __init__(self, factors: List[float]):
        if self.identity_param not in factors:
            factors = [self.identity_param] + list(factors)
        super().__init__("factor", factors)

    def apply_aug_image(self, image, factor=1, **kwargs):
        if factor != self.identity_param:
            image = F.multiply(image, factor)
        return image
