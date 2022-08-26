
import torch
import torch.optim as optim


def get_optimizer(cfg, model_parameter):
    if cfg.optimizer == 'sgd':
        optimizer = optim.SGD(model_parameter, lr=cfg.lr,
                              momentum=0.99, weight_decay=cfg.l2_penalty)
    elif cfg.optimizer == 'adam':
        optimizer = optim.Adam(model_parameter, lr=cfg.lr,
                               betas=(0.9, 0.99), weight_decay=cfg.l2_penalty)
    elif cfg.optimizer == 'adamW':
        optimizer = optim.AdamW(model_parameter, lr=cfg.lr,
                                betas=(0.9, 0.99), weight_decay=cfg.l2_penalty)
    else:
        raise NotImplementedError(f"{cfg.optimizer} method isn't implemented.")

    return optimizer

