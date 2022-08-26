
from .metric import DiceMetric


def get_metric(metric, activation, reduction):
    if metric == 'dice':
        dice_metric_func = DiceMetric(activation=activation, reduction=reduction, eps=1e-8)
    else:
        raise NotImplementedError(f"{metric} metric isn't implemented!")

    return dice_metric_func