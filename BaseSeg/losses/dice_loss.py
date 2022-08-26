
from typing import List

import torch
from torch.nn.modules.loss import _Loss

from .registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class DiceLoss(_Loss):
    def __init__(self, smooth_factor=1e-8, squared_pred=True, label_weighted=None, reduction='mean'):
        """
        :param smooth_factor: a small constant to avoid zeros or nan.
        :param squared_pred: use squared versions of targets and predictions in the denominator or not.
        :param label_weighted: weighted dice for multi-label.
        :param reduction: ['mean', 'sum', 'None'], default to 'mean'.
        """
        super(DiceLoss, self).__init__(reduction)
        self.smooth_factor = smooth_factor
        self.squared_pred = squared_pred
        self.label_weighted = label_weighted

    def forward(self, predict: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        :param predict: shape: [batch, channel, ...]
        :param gt:  shape: [batch, channel, ...]
        :return:
        """
        if len(predict.shape) != len(gt.shape):
            raise ValueError("the number of dimensions for input and target should be the same.")
        reduce_axis: List[int] = torch.arange(2, len(predict.shape)).tolist()

        intersection = torch.sum(predict * gt, dim=reduce_axis)
        if self.squared_pred:
            predict = torch.pow(predict, 2)
            gt = torch.pow(gt, 2)
        union = torch.sum(predict, dim=reduce_axis) + torch.sum(gt, dim=reduce_axis)
        dice_loss = (2. * intersection + self.smooth_factor) / (union + self.smooth_factor)
        dice_loss = 1 - dice_loss

        # perform on batch and channel dims.
        if self.reduction == 'mean':
            dice_loss = dice_loss.mean()
        elif self.reduction == 'sum':
            dice_loss = dice_loss.sum()

        return dice_loss


@LOSS_REGISTRY.register()
class WeightedDiceLoss(_Loss):
    def __init__(self, smooth_factor=1e-8, squared_pred=True, label_weighted=None, reduction='mean'):
        """
        :param smooth_factor: a small constant to avoid zeros or nan.
        :param squared_pred: use squared versions of targets and predictions in the denominator or not.
        :param label_weighted: weighted dice for multi-label.
        :param reduction: ['mean', 'sum', 'None'], default to 'mean'.
        """
        super(WeightedDiceLoss, self).__init__(reduction)
        self.smooth_factor = smooth_factor
        self.squared_pred = squared_pred
        self.label_weighted = [1, 1, 1, 2, 1, 2, 3, 3, 3, 3, 1, 3, 1]

    def forward(self, predict: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        :param predict: shape: [batch, channel, ...]
        :param gt:  shape: [batch, channel, ...]
        :return:
        """
        if len(predict.shape) != len(gt.shape):
            raise ValueError("the number of dimensions for input and target should be the same.")
        reduce_axis: List[int] = torch.arange(2, len(predict.shape)).tolist()

        intersection = torch.sum(predict * gt, dim=reduce_axis)
        if self.squared_pred:
            predict = torch.pow(predict, 2)
            gt = torch.pow(gt, 2)
        union = torch.sum(predict, dim=reduce_axis) + torch.sum(gt, dim=reduce_axis)
        dice_loss = (2. * intersection + self.smooth_factor) / (union + self.smooth_factor)
        dice_loss = 1 - dice_loss

        assert len(self.label_weighted) == predict.shape[1]
        dice_loss = dice_loss*torch.tensor(self.label_weighted).expand(
            predict.shape[0], predict.shape[1]).float().to(gt.device)

        # perform on batch and channel dims.
        if self.reduction == 'mean':
            dice_loss = dice_loss.mean()
        elif self.reduction == 'sum':
            dice_loss = dice_loss.sum()

        return dice_loss