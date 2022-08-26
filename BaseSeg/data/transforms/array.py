"""
Reference to 'https://github.com/HiLab-git/SSL4MIS'
"""

import copy
import random
import numpy as np
from scipy.special import comb

import torch

from Common.transforms.image_resample import ScipyResample


__all__ = ['Resize', 'CenterCrop', 'RandomCrop', 'RandomRotFlip', 'NonlinearTransform',
           'LocalPixelShuffling', 'ImagePainting', 'RandomNoise', 'Normalization', 'ToTensor']


class Resize(object):
    def __init__(self, scales=(0.9, 1.1), p=0.5):
        self.scales = scales
        self.p = p

    def __call__(self, image):
        if random.random() >= self.p:
            return image
        scale = np.random.uniform(self.scales[0], self.scales[1])
        image_shape = image.shape
        out_shape = [int(item*scale) for item in image_shape]
        out_image, _ = ScipyResample.resample_to_size(image, out_shape)

        return out_image


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image):
        image_shape = image.shape
        if image_shape[0] == self.output_size[0] and \
           image_shape[1] == self.output_size[1] and \
           image_shape[2] == self.output_size[2]:
            return image

        # pad the sample if necessary
        if image_shape[0] <= self.output_size[0] or image_shape[1] <= self.output_size[1] \
                or image_shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - image_shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image_shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image_shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)

        (w, h, d) = image.shape
        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        image = image[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]].copy()

        return image


class RandomCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image):
        image_shape = image.shape
        if image_shape[0] == self.output_size[0] and \
           image_shape[1] == self.output_size[1] and \
           image_shape[2] == self.output_size[2]:
            return image

        # pad the sample if necessary
        if image_shape[0] <= self.output_size[0] or image_shape[1] <= self.output_size[1] or \
                image_shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - image_shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image_shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image_shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)

        (w, h, d) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        image = image[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]].copy()

        return image


class RandomRotFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            k = np.random.randint(0, 4)
            # axis = [(0, 1), (1, 2), (0, 2)]
            # idx = np.random.choice([0, 1, 2])
            image = np.rot90(image, k, (1, 2))
        if random.random() < self.p:
            axis = np.random.randint(0, 3)
            image = np.flip(image, axis=axis).copy()

        return image


class NonlinearTransform(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if random.random() >= self.p:
            return image
        image = image[np.newaxis, ]
        points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
        xvals, yvals = self._bezier_curve(points, nTimes=100000)
        if random.random() < 0.5:
            # Half change to get flip
            xvals = np.sort(xvals)
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)
        image = np.interp(image, xvals, yvals)
        image = np.squeeze(image, axis=0)

        return image

    def _bezier_curve(self, points, nTimes=1000):
        """
           Given a set of control points, return the
           bezier curve defined by the control points.

           Control points should be a list of lists, or list of tuples
           such as [ [1,1],
                     [2,3],
                     [4,5], ..[Xn, Yn] ]
            nTimes is the number of time steps, defaults to 1000

            See http://processingjs.nihongoresources.com/bezierinfo/
        """

        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])

        t = np.linspace(0.0, 1.0, nTimes)

        polynomial_array = np.array([self._bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)

        return xvals, yvals

    def _bernstein_poly(self, i, n, t):
        """
         The Bernstein polynomial of n, i as a function of t
        """

        return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


class LocalPixelShuffling(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if random.random() >= self.p:
            return x
        x = x[np.newaxis, ]
        image_temp = copy.deepcopy(x)
        orig_image = copy.deepcopy(x)
        _, img_rows, img_cols, img_deps = x.shape
        num_block = 10000
        for _ in range(num_block):
            block_noise_size_x = random.randint(1, img_rows // 10)
            block_noise_size_y = random.randint(1, img_cols // 10)
            block_noise_size_z = random.randint(1, img_deps // 10)
            noise_x = random.randint(0, img_rows - block_noise_size_x)
            noise_y = random.randint(0, img_cols - block_noise_size_y)
            noise_z = random.randint(0, img_deps - block_noise_size_z)
            window = orig_image[0, noise_x:noise_x + block_noise_size_x,
                                noise_y:noise_y + block_noise_size_y,
                                noise_z:noise_z + block_noise_size_z]
            window = window.flatten()
            np.random.shuffle(window)
            window = window.reshape((block_noise_size_x,
                                     block_noise_size_y,
                                     block_noise_size_z))
            image_temp[0, noise_x:noise_x + block_noise_size_x,
            noise_y:noise_y + block_noise_size_y,
            noise_z:noise_z + block_noise_size_z] = window
        image_temp = np.squeeze(image_temp, axis=0)

        return image_temp


class ImagePainting(object):
    def __init__(self, p=0.5, in_painting_p=0.5):
        self.p = p
        self.in_painting_p = in_painting_p

    def __call__(self, x):
        if random.random() >= self.p:
            return x
        x = x[np.newaxis, ]
        if random.random() < self.in_painting_p:
            x = self._in_painting(x)
        else:
            x = self._out_painting(x)
        x = np.squeeze(x, axis=0)

        return x

    def _in_painting(self, x):
        _, img_rows, img_cols, img_deps = x.shape
        cnt = 5
        while cnt > 0 and random.random() < 0.95:
            block_noise_size_x = random.randint(img_rows // 6, img_rows // 3)
            block_noise_size_y = random.randint(img_cols // 6, img_cols // 3)
            block_noise_size_z = random.randint(img_deps // 6, img_deps // 3)
            noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
            noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
            x[:,
            noise_x:noise_x + block_noise_size_x,
            noise_y:noise_y + block_noise_size_y,
            noise_z:noise_z + block_noise_size_z] = np.random.rand(block_noise_size_x,
                                                                   block_noise_size_y,
                                                                   block_noise_size_z, ) * 1.0
            cnt -= 1
        return x

    def _out_painting(self, x):
        _, img_rows, img_cols, img_deps = x.shape
        image_temp = copy.deepcopy(x)
        x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], ) * 1.0
        block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)
        block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)
        block_noise_size_z = img_deps - random.randint(3 * img_deps // 7, 4 * img_deps // 7)
        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
        noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
        x[:,
        noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y,
        noise_z:noise_z + block_noise_size_z] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                                noise_y:noise_y + block_noise_size_y,
                                                noise_z:noise_z + block_noise_size_z]
        cnt = 4
        while cnt > 0 and random.random() < 0.95:
            block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)
            block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)
            block_noise_size_z = img_deps - random.randint(3 * img_deps // 7, 4 * img_deps // 7)
            noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
            noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
            x[:,
            noise_x:noise_x + block_noise_size_x,
            noise_y:noise_y + block_noise_size_y,
            noise_z:noise_z + block_noise_size_z] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                                               noise_y:noise_y + block_noise_size_y,
                                                               noise_z:noise_z + block_noise_size_z]
            cnt -= 1

        return x


class RandomNoise(object):
    def __init__(self, noise_variance=(0, 0.25), p=0.5):
        self.noise_variance = noise_variance
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            if self.noise_variance[0] == self.noise_variance[1]:
                variance = self.noise_variance[0]
            else:
                variance = random.uniform(self.noise_variance[0], self.noise_variance[1])
            image += np.random.normal(0.0, variance, size=image.shape)
            return image
        else:
            return image


class Normalization(object):
    def __init__(self, window_level):
        self.window_level = window_level

    def __call__(self, image):
        image = np.clip(image,  self.window_level[0],  self.window_level[1])
        image = (image - np.mean(image)) / (np.std(image) + 1e-5)
        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        image = image[np.newaxis, ]
        return torch.from_numpy(image).float()