
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from .registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class FocalLoss(_Loss):
    def __init__(self, alpha=0.5, gamma=2, reduction='None'):
        """
        :param alpha: alpha factor of focal loss.
        :param gamma: gamma factor of focal loss.
        :param reduction: ['sum', None]
        """
        super(FocalLoss, self).__init__(reduction)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predicts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param predicts: shape: [batch, channel, ...]
        :param targets:  shape: [batch, channel, ...]
        :return:
        """
        if len(predicts.shape) != len(targets.shape):
            raise ValueError("the number of dimensions for input and target should be the same.")
        predicts = predicts.float()
        targets = targets.float()
        batch = predicts.shape[0]
        predicts = torch.clamp(predicts, 1e-4, 1.0 - 1e-4)
        alpha_factor = torch.ones(targets.shape).to(targets.device) * self.alpha

        alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
        focal_weight = torch.where(torch.eq(targets, 1.), 1. - predicts, predicts)
        focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

        bce = F.binary_cross_entropy(predicts, targets)
        cls_loss = focal_weight * bce

        cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
        cls_loss = cls_loss.sum() / torch.clamp(torch.sum(targets).float(), min=1.0)
        if self.reduction == 'sum':
            cls_loss = cls_loss * batch

        return cls_loss