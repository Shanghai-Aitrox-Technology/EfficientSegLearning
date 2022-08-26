
import torch
import torch.nn as nn
from torch.nn import functional as F

from .builder import NETWORK_REGISTRY
from ...utils.registy import Registry


__all__ = ['DeployUNet']
deploy_base_layer = Registry('deploy base layer')


class ConvINReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, p=0.2):
        super(ConvINReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.drop = nn.Dropout3d(p=p, inplace=True)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.relu(x)

        return x


class ConvIN3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super(ConvIN3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


@deploy_base_layer.register()
class ResTwoLayerConvBlock(nn.Module):
    def __init__(self, in_channel, inter_channel, out_channel, p=0.2, stride=1):
        """residual block, including two layer convolution, instance normalization, drop out and ReLU"""
        super(ResTwoLayerConvBlock, self).__init__()
        self.residual_unit = nn.Sequential(
            ConvINReLU3D(in_channel, inter_channel, 3, stride=stride, padding=1, p=p),
            ConvIN3D(inter_channel, out_channel, 3, stride=1, padding=1))
        self.shortcut_unit = ConvIN3D(in_channel, out_channel, 1, stride=stride, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.residual_unit(x)
        output += self.shortcut_unit(x)
        output = self.relu(output)

        return output


@deploy_base_layer.register()
class ResFourLayerConvBlock(nn.Module):
    def __init__(self, in_channel, inter_channel, out_channel, p=0.2, stride=1):
        """residual block, including four layer convolution, instance normalization, drop out and ReLU"""
        super(ResFourLayerConvBlock, self).__init__()
        self.residual_unit_1 = nn.Sequential(
            ConvINReLU3D(in_channel, inter_channel, 3, stride=stride, padding=1, p=p),
            ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1))
        self.residual_unit_2 = nn.Sequential(
            ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=p),
            ConvIN3D(inter_channel, out_channel, 3, stride=1, padding=1))
        self.shortcut_unit_1 = ConvIN3D(in_channel, inter_channel, 1, stride=stride, padding=0)
        self.shortcut_unit_2 = nn.Sequential()
        self.relu_1 = nn.ReLU(inplace=True)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x):
        output_1 = self.residual_unit_1(x)
        output_1 += self.shortcut_unit_1(x)
        output_1 = self.relu_1(output_1)

        output_2 = self.residual_unit_2(output_1)
        output_2 += self.shortcut_unit_2(output_1)
        output_2 = self.relu_2(output_2)

        return output_2


class InputLayer(nn.Module):
    """Input layer, including re-sample, clip and normalization image."""

    def __init__(self, input_size, clip_window):
        super(InputLayer, self).__init__()
        self.input_size = input_size
        self.clip_window = clip_window

    def forward(self, x):
        x = F.interpolate(x, size=self.input_size, mode='trilinear', align_corners=True)
        x = torch.clamp(x, min=self.clip_window[0], max=self.clip_window[1])
        mean = torch.mean(x)
        std = torch.std(x)
        x = (x - mean) / (1e-5 + std)
        return x


class OutputLayer(nn.Module):
    """Output layer, re-sample image to original size."""

    def __init__(self):
        super(OutputLayer, self).__init__()

    def forward(self, x, x_input):
        x = F.interpolate(x, size=(x_input.size(2), x_input.size(3), x_input.size(4)), mode='trilinear', align_corners=True)

        return x


@NETWORK_REGISTRY.register()
class DeployUNet(nn.Module):

    def __init__(self, cfg=None):
        super().__init__()

        # UNet parameter.
        num_class = cfg.num_class
        num_channel = cfg.num_channel
        self.is_preprocess = cfg.is_preprocess
        self.is_postprocess = cfg.is_postprocess

        encoder_conv_block = deploy_base_layer.get(cfg.encoder_conv_block)
        decoder_conv_block = deploy_base_layer.get(cfg.decoder_conv_block)

        self.input = InputLayer(input_size=cfg.input_size, clip_window=cfg.clip_window)
        self.output = OutputLayer()

        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv0_0 = encoder_conv_block(1, num_channel[0], num_channel[0])
        self.conv1_0 = encoder_conv_block(num_channel[0], num_channel[1], num_channel[1])
        self.conv2_0 = encoder_conv_block(num_channel[1], num_channel[2], num_channel[2])
        self.conv3_0 = encoder_conv_block(num_channel[2], num_channel[3], num_channel[3])
        self.conv4_0 = encoder_conv_block(num_channel[3], num_channel[4], num_channel[4])

        self.conv3_1 = decoder_conv_block(num_channel[3] + num_channel[4], num_channel[3], num_channel[3])
        self.conv2_2 = decoder_conv_block(num_channel[2] + num_channel[3], num_channel[2], num_channel[2])
        self.conv1_3 = decoder_conv_block(num_channel[1] + num_channel[2], num_channel[1], num_channel[1])
        self.conv0_4 = decoder_conv_block(num_channel[0] + num_channel[1], num_channel[0], num_channel[0])

        self.final = nn.Conv3d(num_channel[0], num_class, kernel_size=1, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x_input = x
        if self.is_preprocess:
            x = self.input(x)

        x = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_0 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_0 = self.conv2_2(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_0 = self.conv1_3(torch.cat([x1_0, self.up(x2_0)], 1))
        x = self.conv0_4(torch.cat([x, self.up(x1_0)], 1))
        x = self.final(x)
        if self.is_postprocess:
            x = self.output(x, x_input)
        x = F.sigmoid(x)

        return x