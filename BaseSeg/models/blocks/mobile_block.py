
import torch
import torch.nn as nn

from .builder import BLOCK_REGISTRY


class ConvINReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvINReLU, self).__init__(
            nn.Conv3d(in_channel, out_channel, (kernel_size, kernel_size, kernel_size),
                      (stride, stride, stride), (padding, padding, padding), groups=groups, bias=False),
            nn.InstanceNorm3d(out_channel),
            nn.LeakyReLU(inplace=True)
        )


@BLOCK_REGISTRY.register()
class MobileInvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(MobileInvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvINReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvINReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv3d(hidden_channel, out_channel, kernel_size=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


@BLOCK_REGISTRY.register()
class MobileFeatureBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(MobileFeatureBlock, self).__init__()
        self.stride2_conv = MobileInvertedResidual(in_channel, out_channel, stride=stride, expand_ratio=expand_ratio)
        self.stride1_conv = MobileInvertedResidual(out_channel, out_channel, stride=1, expand_ratio=expand_ratio)

    def forward(self, x):
        x = self.stride2_conv(x)
        x = self.stride1_conv(x)

        return x