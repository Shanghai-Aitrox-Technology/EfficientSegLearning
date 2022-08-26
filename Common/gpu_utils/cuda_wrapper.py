import torch
from functools import wraps


def cuda_wrapper_model(model, is_cuda: bool, is_half: bool, mode: str='val'):
    if not is_cuda:
        is_half = False
    if is_cuda:
        model = model.cuda()
        if is_half:
            model = model.half()
    if mode == 'train':
        model = model.train()
    else:
        model = model.eval()

    return model


def cuda_wrapper_data(data, is_cuda: bool, is_half: bool):
    if not is_cuda:
        is_half = False
    if is_cuda:
        data = data.cuda()
        if is_half:
            data = data.half()

    return data


