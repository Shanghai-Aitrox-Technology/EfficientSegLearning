
import os
from typing import Collection, Hashable, Iterable, Sequence, TypeVar, Union

import numpy as np
import torch

__all__ = [
    "KeysCollection",
    "IndexSelection",
    "DtypeLike",
    "NdarrayTensor",
    "NdarrayOrTensor",
    "TensorOrList",
    "PathLike",
]


KeysCollection = Union[Collection[Hashable], Hashable]
IndexSelection = Union[Iterable[int], int]
DtypeLike = Union[np.dtype, type, None]
NdarrayTensor = TypeVar("NdarrayTensor", np.ndarray, torch.Tensor)
NdarrayOrTensor = Union[np.ndarray, torch.Tensor]
TensorOrList = Union[torch.Tensor, Sequence[torch.Tensor]]
PathLike = Union[str, os.PathLike]
