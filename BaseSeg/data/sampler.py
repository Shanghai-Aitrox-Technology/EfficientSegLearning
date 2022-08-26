"""
Reference to "MONAI"
"""

import itertools
import numpy as np
from typing import Optional, Sequence
from torch.utils.data.sampler import Sampler


import torch
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler as _TorchDistributedSampler

__all__ = ["DistributedSampler", "DistributedWeightedRandomSampler", "TwoStreamBatchSampler"]


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


class DistributedSampler(_TorchDistributedSampler):
    """
    Enhance PyTorch DistributedSampler to support non-evenly divisible sampling.

    Args:
        dataset: Dataset used for sampling.
        even_divisible: if False, different ranks can have different data length.
            for example, input data: [1, 2, 3, 4, 5], rank 0: [1, 3, 5], rank 1: [2, 4].
        num_replicas: number of processes participating in distributed training.
            by default, `world_size` is retrieved from the current distributed group.
        rank: rank of the current process within `num_replicas`. by default,
            `rank` is retrieved from the current distributed group.
        shuffle: if `True`, sampler will shuffle the indices, default to True.
        kwargs: additional arguments for `DistributedSampler` super class, can be `seed` and `drop_last`.

    More information about DistributedSampler, please check:
    https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler.

    """

    def __init__(
        self,
        dataset: Dataset,
        even_divisible: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        **kwargs,
    ):
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, **kwargs)

        if not even_divisible:
            data_len = len(dataset)  # type: ignore
            extra_size = self.total_size - data_len
            if self.rank + extra_size >= self.num_replicas:
                self.num_samples -= 1
            self.total_size = data_len


class DistributedWeightedRandomSampler(DistributedSampler):
    """
    Extend the `DistributedSampler` to support weighted sampling.
    Refer to `torch.utils.data.WeightedRandomSampler`, for more details please check:
    https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler.

    Args:
        dataset: Dataset used for sampling.
        weights: a sequence of weights, not necessary summing up to one, length should exactly
            match the full dataset.
        num_samples_per_rank: number of samples to draw for every rank, sample from
            the distributed subset of dataset.
            if None, default to the length of dataset split by DistributedSampler.
        generator: PyTorch Generator used in sampling.
        even_divisible: if False, different ranks can have different data length.
            for example, input data: [1, 2, 3, 4, 5], rank 0: [1, 3, 5], rank 1: [2, 4].'
        num_replicas: number of processes participating in distributed training.
            by default, `world_size` is retrieved from the current distributed group.
        rank: rank of the current process within `num_replicas`. by default,
            `rank` is retrieved from the current distributed group.
        shuffle: if `True`, sampler will shuffle the indices, default to True.
        kwargs: additional arguments for `DistributedSampler` super class, can be `seed` and `drop_last`.

    """

    def __init__(
        self,
        dataset: Dataset,
        weights: Sequence[float],
        num_samples_per_rank: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        even_divisible: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            even_divisible=even_divisible,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            **kwargs,
        )
        self.weights = weights
        self.num_samples_per_rank = num_samples_per_rank if num_samples_per_rank is not None else self.num_samples
        self.generator = generator

    def __iter__(self):
        indices = list(super().__iter__())
        weights = torch.as_tensor([self.weights[i] for i in indices], dtype=torch.double)
        # sample based on the provided weights
        rand_tensor = torch.multinomial(weights, self.num_samples_per_rank, True, generator=self.generator)

        for i in rand_tensor:
            yield indices[i]

    def __len__(self):
        return self.num_samples_per_rank
