
from addict import Dict
import torch
from BaseSeg.models.network.builder import NETWORK_REGISTRY
from Common.utils.config import Config


__all__ = ['get_network']


def get_network(cfg):
    model = NETWORK_REGISTRY.get(cfg.meta_architecture)(cfg)

    return model


if __name__ == '__main__':
    data = torch.randn([1, 1, 192, 192, 192]).float().cuda()
    model_cfg = {
        'meta_architecture': 'EfficientSegNet',
        'num_channel': [16, 32, 64, 128, 256],
        'num_class': 4,
        'num_depth': 4,
        'num_blocks': [2, 2, 2, 2],
        'decoder_num_block': 2,
        'auxiliary_task': False,
        'auxiliary_class': 1,
        'encoder_conv_block': 'AnisotropicConvBlock',
        'decoder_conv_block': 'AnisotropicConvBlock',
        'context_block': 'AnisotropicAvgPooling',
        'input_size': [160, 160, 160],
        'clip_window': [-325, 325],
        'is_preprocess': True,
        'is_postprocess': False,
        'is_dynamic_empty_cache': True
    }
    # model_cfg = Dict(model_cfg)
    # model = get_network(model_cfg).cuda()

    config_path = './default_model_config.yaml'
    default_cfg = Config.fromfile(config_path)
    default_cfg.merge_from_dict(model_cfg)
    model = get_network(default_cfg.efficientSegNet).cuda()
    print(default_cfg.pretty_text)

    with torch.no_grad():
        outputs = model(data)

    print(outputs.shape)