
import torch
import torch.nn.functional as F


class Registry(object):

    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name

        self._obj_map = {}

    def get_obj_name(self):

        return self._obj_map.keys()

    def _do_register(self, name, obj):
        assert (
            name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(name, self._name)
        self._obj_map[name] = obj

    def register(self, obj=None):
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError("No object named '{}' found in '{}' registry!".format(name, self._name))
        return ret


def rot90(x, k=1, axis=(2, 3)):
    """rotate batch of images by 90 degrees k times along the axis"""
    return torch.rot90(x, k, axis)


def flip(x, axis=0):
    """flip batch of images along the axis"""
    return x.flip(axis)


def sum(x1, x2):
    """sum of two tensors"""
    return x1 + x2


def add(x, value):
    """add value to tensor"""
    return x + value


def max_t(x1, x2):
    """compare 2 tensors and take max values"""
    return torch.max(x1, x2)


def min(x1, x2):
    """compare 2 tensors and take min values"""
    return torch.min(x1, x2)


def multiply(x, factor):
    """multiply tensor by factor"""
    return x * factor


def scale(x, scale_factor, interpolation="trilinear", align_corners=True):
    """scale batch of images by `scale_factor` with given interpolation mode"""
    size = x.shape[2:]
    image_size = [int(scale_factor*item) for item in size]
    return F.interpolate(x, size=image_size, mode=interpolation, align_corners=align_corners)


def resize(x, size, interpolation="trilinear", align_corners=True):
    """resize batch of images to given spatial size with given interpolation mode"""
    return F.interpolate(x, size=size, mode=interpolation, align_corners=align_corners)


def crop_pad(x, target_size):
    raw_size = list(x.shape[2:])
    crop_or_pad = 1
    if raw_size[0] <= target_size[0] or raw_size[1] <= target_size[1] or \
            raw_size[2] <= target_size[2]:
        pw = max((target_size[0] - raw_size[0]) // 2 + 3, 0)
        ph = max((target_size[1] - raw_size[1]) // 2 + 3, 0)
        pd = max((target_size[2] - raw_size[2]) // 2 + 3, 0)
        x = F.pad(x, [pd, pd, ph, ph, pw, pw], mode='constant', value=0)
        pad_start = [pw, ph, pd]
        crop_or_pad = 2

    (w, h, d) = x.shape[2:]
    w1 = int(round((w - target_size[0]) / 2.))
    h1 = int(round((h - target_size[1]) / 2.))
    d1 = int(round((d - target_size[2]) / 2.))

    x = x[:, :, w1:w1 + target_size[0], h1:h1 + target_size[1], d1:d1 + target_size[2]]
    if crop_or_pad == 1:
        target_bbox = [[w1, w1+target_size[0]],
                       [h1, h1+target_size[1]],
                       [d1, d1+target_size[2]]]
    else:
        target_bbox = [[pad_start[0]-w1, pad_start[0]-w1+raw_size[0]],
                       [pad_start[1]-h1, pad_start[1]-h1+raw_size[1]],
                       [pad_start[2]-d1, pad_start[2]-d1+raw_size[2]]]

    return x, target_bbox, crop_or_pad


def crop_sss(x, crop_h, crop_w, crop_d):
    """crop h_start w_start d_start"""
    crop_bbox = [[0, crop_h], [0, crop_w], [0, crop_d]]

    return crop_bbox


def crop_eee(x, crop_h, crop_w, crop_d):
    """crop h_end w_end d_end"""
    image_shape = x.shape
    crop_bbox = [[image_shape[2]-crop_h, image_shape[2]],
                 [image_shape[3]-crop_w, image_shape[3]],
                 [image_shape[4]-crop_d, image_shape[4]]]
    return crop_bbox


def crop_ess(x, crop_h, crop_w, crop_d):
    image_shape = x.shape
    crop_bbox = [[image_shape[2]-crop_h, image_shape[2]],
                 [0, crop_w],
                 [0, crop_d]]
    return crop_bbox


def crop_ses(x, crop_h, crop_w, crop_d):
    image_shape = x.shape
    crop_bbox = [[0, crop_h],
                 [image_shape[3]-crop_w, image_shape[3]],
                 [0, crop_d]]
    return crop_bbox


def crop_sse(x, crop_h, crop_w, crop_d):
    image_shape = x.shape
    crop_bbox = [[0, crop_h],
                 [0, crop_w],
                 [image_shape[4]-crop_d, image_shape[4]]]
    return crop_bbox


def crop_see(x, crop_h, crop_w, crop_d):
    image_shape = x.shape
    crop_bbox = [[0, crop_h],
                 [image_shape[3]-crop_w, image_shape[3]],
                 [image_shape[4]-crop_d, image_shape[4]]]
    return crop_bbox


def crop_ese(x, crop_h, crop_w, crop_d):
    image_shape = x.shape
    crop_bbox = [[image_shape[2]-crop_h, image_shape[2]],
                 [0, crop_w],
                 [image_shape[4]-crop_d, image_shape[4]]]
    return crop_bbox


def crop_ees(x, crop_h, crop_w, crop_d):
    image_shape = x.shape
    crop_bbox = [[image_shape[2]-crop_h, image_shape[2]],
                 [image_shape[3]-crop_w, image_shape[3]],
                 [0, crop_d]]
    return crop_bbox


def crop_ccc(x, crop_h, crop_w, crop_d):
    """make center crop"""

    center_h = x.shape[2] // 2
    center_w = x.shape[3] // 2
    center_d = x.shape[4] // 2
    half_crop_h = crop_h // 2
    half_crop_w = crop_w // 2
    half_crop_d = crop_d // 2

    y_min = center_h - half_crop_h
    y_max = center_h + half_crop_h + crop_h % 2
    x_min = center_w - half_crop_w
    x_max = center_w + half_crop_w + crop_w % 2
    z_min = center_d - half_crop_d
    z_max = center_d + half_crop_d + crop_d % 2

    crop_bbox = [[y_min, y_max], [x_min, x_max], [z_min, z_max]]

    return crop_bbox


def crop_zzz(x, crop_h, crop_w, crop_d):
    image_shape = x.shape[2:]
    crop_bbox = [[0, image_shape[0]], [0, image_shape[1]], [0, image_shape[2]]]

    return crop_bbox


crop_registry = Registry('crop_registry')
crop_registry.register(crop_sss)
crop_registry.register(crop_eee)
crop_registry.register(crop_ess)
crop_registry.register(crop_ses)
crop_registry.register(crop_sse)
crop_registry.register(crop_see)
crop_registry.register(crop_ese)
crop_registry.register(crop_ees)
crop_registry.register(crop_ccc)
crop_registry.register(crop_zzz)