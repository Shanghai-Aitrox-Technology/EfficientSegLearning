import itertools
from functools import partial
from typing import List, Union, Optional


import torch
from . import functional as F


class BaseTransform:
    identity_param = None

    def __init__(
            self,
            name: str = 'transform',
            params: Optional[Union[list, tuple]] = None,
    ):
        self.pname = name
        self.params = params

    def apply_aug_image(self, image, *args, **params):
        raise NotImplementedError

    def apply_deaug_mask(self, mask, *args, **params):
        raise NotImplementedError


class ImageOnlyTransform(BaseTransform):
    def apply_deaug_mask(self, mask, *args, **params):
        return mask


class DualTransform(BaseTransform):
    pass


class Chain:

    def __init__(
            self,
            functions: List[callable]
    ):
        self.functions = functions or []

    def __call__(self, x):
        for f in self.functions:
            x = f(x)
        return x


class Transformer:
    def __init__(
            self,
            image_pipeline: Chain,
            mask_pipeline: Chain
    ):
        self.image_pipeline = image_pipeline
        self.mask_pipeline = mask_pipeline

    def augment_image(self, image):
        return self.image_pipeline(image)

    def deaugment_mask(self, mask):
        return self.mask_pipeline(mask)


class Compose:

    def __init__(
            self,
            transforms: List[BaseTransform],
    ):
        self.aug_transforms = transforms
        self.aug_transform_parameters = list(itertools.product(*[t.params for t in self.aug_transforms]))
        self.deaug_transforms = transforms[::-1]
        self.deaug_transform_parameters = [p[::-1] for p in self.aug_transform_parameters]

    def __iter__(self) -> Transformer:
        for aug_params, deaug_params in zip(self.aug_transform_parameters, self.deaug_transform_parameters):
            image_aug_chain = Chain([partial(t.apply_aug_image, **{t.pname: p})
                                     for t, p in zip(self.aug_transforms, aug_params)])
            mask_deaug_chain = Chain([partial(t.apply_deaug_mask, **{t.pname: p})
                                      for t, p in zip(self.deaug_transforms, deaug_params)])
            yield Transformer(
                image_pipeline=image_aug_chain,
                mask_pipeline=mask_deaug_chain
            )

    def __len__(self) -> int:
        return len(self.aug_transform_parameters)


class Merger:
    def __init__(self, raw_shape, type='mean'):
        if type not in ['mean']:
            raise ValueError('Not correct merge type `{}`.'.format(type))

        self.type = type
        self.output = torch.zeros(raw_shape).float()
        self.count_matrix = torch.zeros(raw_shape).float()

    def append(self, x, target_bbox):
        if self.type == 'mean':
            self.output[:, :, target_bbox[0][0]:target_bbox[0][1],
                        target_bbox[1][0]:target_bbox[1][1],
                        target_bbox[2][0]:target_bbox[2][1]] += x
            self.count_matrix[:, :, target_bbox[0][0]:target_bbox[0][1],
                              target_bbox[1][0]:target_bbox[1][1],
                              target_bbox[2][0]:target_bbox[2][1]] += 1

    @property
    def result(self):
        if self.type == 'mean':
            output = self.output / self.count_matrix
        else:
            output = self.output

        return output
