
from typing import Sequence

import torch
from torch.optim import Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


def get_lr_scheduler(cfg, optimizer):
    if cfg.lr_schedule == "cosineLR":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=3e-5)
    elif cfg.lr_schedule == "stepLR":
        return optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.5)
    elif cfg.lr_schedule == "poly":
        raise NotImplementedError(f"{cfg.lr_schedule} method isn't implemented.")
    else:
        raise NotImplementedError(f"{cfg.lr_schedule} method isn't implemented.")


class CustomStepLR(_LRScheduler):
    def __init__(self, optimizer, step_epoch: Sequence[int], gamma: Sequence[float],
                 last_epoch: int = -1, verbose: bool = False):
        """
        Args:
        :param optimizer:
        :param step_epoch:
        :param gamma:
        :param last_epoch:
        :param verbose:

        Examples:
            num_epochs = 200
            step_epoch = [int(num_epochs * 0.66), int(num_epochs * 0.86), num_epochs]
            gamma = [1.0, 0.1, 0.05]
        """
        if len(step_epoch) != len(gamma):
            raise ValueError(f'length of step_epoch and gamma must be the same!')
        self.step_epoch = step_epoch
        self.gamma = gamma
        super(CustomStepLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        cur_index = 0
        for i in range(len(self.step_epoch)):
            if self.last_epoch < self.step_epoch[i]:
                cur_index = i
                break
        cur_gamma = self.gamma[cur_index]
        return [group['initial_lr'] * cur_gamma
                for group in self.optimizer.param_groups]


class WarmupPolyLR(LambdaLR):
    def __init__(self, optimizer: Optimizer, num_step: int, total_epoch: int, is_warmup: bool = True,
                 warmup_epoch: int = 1, warmup_factor: float = 1e-3, last_epoch: int = -1,
                 verbose: bool = False) -> None:
        """ """
        assert num_step > 0 and total_epoch > 0

        if is_warmup is False:
            warmup_epoch = 0
        self.is_warmup = is_warmup
        self.warmup_epoch = warmup_epoch
        self.num_step = num_step
        self.total_epoch = total_epoch
        self.warmup_factor = warmup_factor
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if self.is_warmup and step <= (self.warmup_epoch * self.num_step):
            alpha = float(step) / (self.warmup_epoch * self.num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return self.warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            # lr*(1-iter/max_iter)^power
            return (1 - (step - self.warmup_epoch * self.num_step) /
                    ((self.total_epoch - self.warmup_epoch) * self.num_step)) ** 0.9


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            # lr*(1-iter/max_iter)^power
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)