from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from .registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class PyramidConsistencyLoss(_Loss):
    def __init__(self, labeled_num=1, activate='sigmoid', reduction='mean'):
        super(PyramidConsistencyLoss, self).__init__(reduction)
        self.labeled_num = labeled_num
        self.activate = activate
        self.reduction = reduction
        self.kl_distance = nn.KLDivLoss(reduction='none')

    def forward(self, predict: List[torch.Tensor]):
        pred_maps = 0
        num = len(predict)
        for i in range(num):
            if self.activate == 'sigmoid':
                pred_maps += F.sigmoid(predict[i][self.labeled_num:])
            else:
                pred_maps += F.softmax(predict[i][self.labeled_num:], dim=1)
        pred_maps /= num

        consistency_loss = 0
        for i in range(num):
            if self.activate == 'sigmoid':
                activate_map = F.sigmoid(predict[i][self.labeled_num:])
            else:
                activate_map = F.softmax(predict[i][self.labeled_num:], dim=1)
            consistency_dist = (pred_maps - activate_map) ** 2
            # variance = torch.sum(self.kl_distance(torch.log(activate_map), pred_maps),
            #                      dim=1, keepdim=True)
            # variance = torch.clamp(variance, 1e-4, 1.0 - 1e-4)
            # exp_variance = torch.exp(-variance)
            # loss = torch.mean(consistency_dist * exp_variance) / \
            #                  (torch.mean(exp_variance) + 1e-8) + torch.mean(variance)
            # consistency_loss += loss
            # del activate_map, consistency_dist, variance, exp_variance, loss
            consistency_loss += torch.mean(consistency_dist)

        consistency_loss /= num

        return consistency_loss
