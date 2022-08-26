import torch
import torch.nn as nn

from .builder import BLOCK_REGISTRY


def channel_shuffle(x, groups: int):

    batch_size, num_channels, height, width, depth = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width, depth] ->
    # [batch_size, groups, channels_per_group, height, width, depth]
    x = x.view(batch_size, groups, channels_per_group, height, width, depth)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width, depth)

    return x


@BLOCK_REGISTRY.register()
class ShuffleInvertedResidual(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int, is_dynamic_empty_cache: bool = True):
        super(ShuffleInvertedResidual, self).__init__()
        self.is_dynamic_empty_cache = is_dynamic_empty_cache

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.stride = stride

        assert output_c % 2 == 0
        branch_features = output_c // 2
        # 当stride为1时，input_channel应该是branch_features的两倍
        # python中 '<<' 是位运算，可理解为计算×2的快速方法
        assert (self.stride != 1) or (input_c == branch_features << 1)

        if self.stride == 2:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                nn.InstanceNorm3d(input_c),
                nn.Conv3d(input_c, branch_features, kernel_size=(1, 1, 1), bias=False),
                nn.InstanceNorm3d(branch_features),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv3d(input_c if self.stride > 1 else branch_features,
                      branch_features, kernel_size=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(branch_features),
            nn.LeakyReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),
            nn.InstanceNorm3d(branch_features),
            nn.Conv3d(branch_features, branch_features, kernel_size=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(branch_features),
            nn.LeakyReLU(inplace=True)
        )

    @staticmethod
    def depthwise_conv(input_c: int,
                       output_c: int,
                       kernel_s: int,
                       stride: int = 1,
                       padding: int = 0,
                       bias: bool = False) -> nn.Conv3d:
        return nn.Conv3d(in_channels=input_c, out_channels=output_c, kernel_size=(kernel_s, kernel_s, kernel_s),
                         stride=(stride, stride, stride), padding=(padding, padding, padding),
                         bias=bias, groups=input_c)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
            if self.is_dynamic_empty_cache:
                del x, x1, x2
                torch.cuda.empty_cache()
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
            if self.is_dynamic_empty_cache:
                del x
                torch.cuda.empty_cache()

        out = channel_shuffle(out, 2)

        return out


@BLOCK_REGISTRY.register()
class ShuffleFeatureBlock(nn.Module):
    def __init__(self, in_channel, middle_channel, out_channel, stride=1, is_dynamic_empty_cache=True):
        super(ShuffleFeatureBlock, self).__init__()
        self.stride2_conv = ShuffleInvertedResidual(in_channel, out_channel, stride=stride,
                                                    is_dynamic_empty_cache=is_dynamic_empty_cache)
        self.stride1_conv = ShuffleInvertedResidual(out_channel, out_channel, stride=1,
                                                    is_dynamic_empty_cache=is_dynamic_empty_cache)

    def forward(self, x):
        x = self.stride2_conv(x)
        x = self.stride1_conv(x)

        return x