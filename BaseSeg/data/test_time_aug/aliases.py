from .base import Compose
from . import transforms as tta


def default_transform():
    return Compose([tta.Flip(axis=[2, 3, 4]),
                    tta.Rotate90(k=[1, 2, 3], axis=(3, 4)),
                    tta.Scale(scales=[1, 2, 4])])
