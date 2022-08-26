"""
Reference: https://github.com/qubvel/ttach

Pipeline of test time augmentation:
# input batch of images
# apply augmentations (flips, rotation, scale, etc.)
# pass augmented batches through model
# reverse transformations for each batch of masks/labels
# merge predictions (mean, max, gmean, etc.)
# output batch of masks/labels

Examples:
import test_time_aug as tta
transforms = tta.Compose([
        tta.Flip(axis=[2, 3, 4]),
        tta.Rotate90(k=[1, 2, 3], axis=(3, 4)),
        tta.Scale(scales=[1, 2, 4])
    ])
tta_model = tta.SegmentationTTAWrapper(model, transforms)
"""

import torch
import torch.nn as nn
from typing import Optional, Mapping, Union

from .base import Merger, Compose


class SegmentationTTAWrapper(nn.Module):
    """Wrap PyTorch nn.Module (segmentation model) with test time augmentation transforms

    Args:
        model (torch.nn.Module): segmentation model with single input and single output
            (.forward(x) should return either torch.Tensor or Mapping[str, torch.Tensor])
        transforms (ttach.Compose): composition of test time transforms
        merge_mode (str): method to merge augmented predictions mean/gmean/max/min/sum/tsharpen
        output_mask_key (str): if model output is `dict`, specify which key belong to `mask`
    """

    def __init__(
        self,
        model: nn.Module,
        transforms: Compose,
        num_class: int = 1,
        merge_mode: str = "mean",
        output_mask_key: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.num_class = num_class
        self.merge_mode = merge_mode
        self.output_key = output_mask_key

    def forward(
        self, image: torch.Tensor, *args
    ) -> Union[torch.Tensor, Mapping[str, torch.Tensor]]:
        raw_shape = image.shape
        out_shape = [raw_shape[0], self.num_class, raw_shape[2], raw_shape[3], raw_shape[4]]
        merger = Merger(raw_shape=out_shape, type=self.merge_mode)

        for transformer in self.transforms:
            augmented_image = transformer.augment_image(image)
            augmented_output = self.model(augmented_image, *args)
            augmented_output = augmented_output.sigmoid_()
            augmented_output = augmented_output.cpu().float()
            if self.output_key is not None:
                augmented_output = augmented_output[self.output_key]
            deaugmented_output, target_bbox = transformer.deaugment_mask(augmented_output)
            merger.append(deaugmented_output, target_bbox)

        result = merger.result
        if self.output_key is not None:
            result = {self.output_key: result}

        return result

