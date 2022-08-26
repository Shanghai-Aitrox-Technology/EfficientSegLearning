
from typing import Sequence, Optional, Union

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from .dice_loss import DiceLoss, WeightedDiceLoss
from .focal_loss import FocalLoss
from .hausdorff_loss import HausdorffLoss
from .registry import LOSS_REGISTRY

_LOSS_FUNC = ['DiceLoss', 'WeightedDiceLoss', 'FocalLoss', 'HausdorffLoss']
_ACTIVATION = ['sigmoid', 'softmax']


class SegLoss(_Loss):
    def __init__(self, loss_func: Sequence[str],
                 loss_weight: Sequence[float],
                 activation: str = 'sigmoid',
                 reduction: str = 'mean',
                 num_label: int = 1):
        """
        :param loss_func: List of loss function.
        :param loss_weight: List of loss weight.
        :param activation: activation function for output.
        :param reduction: ['mean', 'sum', 'None']
        :param num_label: Number of label.
        """
        super(SegLoss, self).__init__(reduction)
        for loss in loss_func:
            if loss not in _LOSS_FUNC:
                raise TypeError(f'{loss} is invalid')
        if activation not in _ACTIVATION:
            raise TypeError(f'{activation} is invalid')
        if len(loss_func) != len(loss_weight):
            raise ValueError(f'loss_func and loss_weight must be the same size!')

        self.loss_func = loss_func
        self.loss_weight = loss_weight
        self.activation = activation
        self.num_label = num_label

    def forward(self, predict, gt):
        """
        :param predict: shape: [batch, channel, ...]
        :param gt:  shape: [batch, channel, ...]
        :return:
        """
        predict = predict.float()
        gt = gt.float()
        if self.activation == 'softmax':
            predict = F.softmax(predict, dim=1)
        elif self.activation == 'sigmoid':
            predict = F.sigmoid(predict)

        loss = 0
        for idx, loss_name in enumerate(self.loss_func):
            loss_func = LOSS_REGISTRY.get(loss_name)(reduction=self.reduction)
            loss += self.loss_weight[idx] * loss_func(predict, gt)

        return loss