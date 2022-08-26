
import torch
import torch.nn as nn
from torch.nn import functional as F

from .builder import NETWORK_REGISTRY
from ...utils.registy import Registry


__all__ = ['DeployEfficientSegNet']
deploy_efficient_layer = Registry('deploy efficient layer')


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


@deploy_efficient_layer.register()
class AnisotropicAvgPooling(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(12, 12, 12)):
        super(AnisotropicAvgPooling, self).__init__()
        self.pool1 = nn.AvgPool3d(kernel_size=(2, 2, 2))
        self.pool2 = nn.AvgPool3d(kernel_size=(4, 4, 4))
        self.pool3 = nn.AvgPool3d(kernel_size=(1, kernel_size[1], kernel_size[2]))
        self.pool4 = nn.AvgPool3d(kernel_size=(kernel_size[0], 1, kernel_size[2]))
        self.pool5 = nn.AvgPool3d(kernel_size=(kernel_size[0], kernel_size[1], 1))

        inter_channel = in_channel // 4

        self.trans_layer = ConvINReLU3D(in_channel, inter_channel, 1, stride=1, padding=0, p=0.2)
        self.conv1_1 = ConvINReLU3D(inter_channel, inter_channel, 1, stride=1, padding=0, p=0.2)
        self.conv1_2 = ConvINReLU3D(inter_channel, inter_channel, 1, stride=1, padding=0, p=0.2)
        self.conv2_0 = ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv2_1 = ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv2_2 = ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv2_3 = ConvIN3D(inter_channel, inter_channel, (3, 1, 1), stride=1, padding=(1, 0, 0))
        self.conv2_4 = ConvIN3D(inter_channel, inter_channel, (1, 3, 1), stride=1, padding=(0, 1, 0))
        self.conv2_5 = ConvIN3D(inter_channel, inter_channel, (1, 1, 3), stride=1, padding=(0, 0, 1))

        self.conv2_6 = ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=0.2)
        self.conv2_7 = ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=0.2)
        self.conv3 = ConvIN3D(inter_channel*2, inter_channel, 1, stride=1, padding=0)
        self.score_layer = nn.Sequential(ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=0.2),
                                         nn.Conv3d(inter_channel, out_channel, 1, bias=False))

    def forward(self, x):
        size = x.size()[2:]
        x0 = self.trans_layer(x)

        x1 = self.conv1_1(x0)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), size, mode='trilinear', align_corners=True)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), size, mode='trilinear', align_corners=True)
        out1 = self.conv2_6(F.relu(x2_1 + x2_2 + x2_3, inplace=True))

        x2 = self.conv1_2(x0)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), size, mode='trilinear', align_corners=True)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), size, mode='trilinear', align_corners=True)
        x2_6 = F.interpolate(self.conv2_5(self.pool5(x2)), size, mode='trilinear', align_corners=True)
        out2 = self.conv2_7(F.relu(x2_4 + x2_5 + x2_6, inplace=True))

        out = self.conv3(torch.cat([out1, out2], dim=1))
        out = F.relu(x0 + out, inplace=True)

        out = self.score_layer(out)

        return out


@deploy_efficient_layer.register()
class AnisotropicMaxPooling(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(12, 12, 12)):
        super(AnisotropicMaxPooling, self).__init__()
        self.pool1 = nn.AvgPool3d(kernel_size=(2, 2, 2))
        self.pool2 = nn.AvgPool3d(kernel_size=(4, 4, 4))
        self.pool3 = nn.MaxPool3d(kernel_size=(kernel_size[0], 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(1, kernel_size[1], 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(1, 1, kernel_size[2]))

        inter_channel = in_channel // 4

        self.trans_layer = ConvINReLU3D(in_channel, inter_channel, 1, stride=1, padding=0, p=0.2)
        self.conv1_1 = ConvINReLU3D(inter_channel, inter_channel, 1, stride=1, padding=0, p=0.2)
        self.conv1_2 = ConvINReLU3D(inter_channel, inter_channel, 1, stride=1, padding=0, p=0.2)
        self.conv2_0 = ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv2_1 = ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv2_2 = ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv2_3 = ConvIN3D(inter_channel, inter_channel, (1, 3, 3), stride=1, padding=(1, 0, 0))
        self.conv2_4 = ConvIN3D(inter_channel, inter_channel, (3, 1, 3), stride=1, padding=(0, 1, 0))
        self.conv2_5 = ConvIN3D(inter_channel, inter_channel, (3, 3, 1), stride=1, padding=(0, 0, 1))

        self.conv2_6 = ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=0.2)
        self.conv2_7 = ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=0.2)
        self.conv3 = ConvIN3D(inter_channel*2, inter_channel, 1, stride=1, padding=0)
        self.score_layer = nn.Sequential(ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=0.2),
                                         nn.Conv3d(inter_channel, out_channel, 1, bias=False))

    def forward(self, x):
        size = x.size()[2:]
        x0 = self.trans_layer(x)

        x1 = self.conv1_1(x0)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), size, mode='trilinear', align_corners=True)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), size, mode='trilinear', align_corners=True)
        out1 = self.conv2_6(F.relu(x2_1 + x2_2 + x2_3, inplace=True))

        x2 = self.conv1_2(x0)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), size, mode='trilinear', align_corners=True)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), size, mode='trilinear', align_corners=True)
        x2_6 = F.interpolate(self.conv2_5(self.pool5(x2)), size, mode='trilinear', align_corners=True)
        out2 = self.conv2_7(F.relu(x2_4 + x2_5 + x2_6, inplace=True))

        out = self.conv3(torch.cat([out1, out2], dim=1))
        out = F.relu(x0 + out, inplace=True)

        out = self.score_layer(out)

        return out


@deploy_efficient_layer.register()
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


@deploy_efficient_layer.register()
class OutputLayer(nn.Module):
    """Output layer, re-sample image to original size."""

    def __init__(self):
        super(OutputLayer, self).__init__()

    def forward(self, x, x_input):
        x = F.interpolate(x, size=(x_input.size(2), x_input.size(3), x_input.size(4)), mode='trilinear', align_corners=True)

        return x


@deploy_efficient_layer.register()
class ResBaseConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, p=0.2, stride=1, is_identify=True):
        """residual base block, including two layer convolution, instance normalization, drop out and leaky ReLU"""
        super(ResBaseConvBlock, self).__init__()
        self.residual_unit = nn.Sequential(
            ConvINReLU3D(in_channel, out_channel, 3, stride=stride, padding=1, p=p),
            ConvIN3D(out_channel, out_channel, 3, stride=1, padding=1))
        self.shortcut_unit = nn.Sequential() if stride == 1 and in_channel == out_channel and is_identify else \
            ConvIN3D(in_channel, out_channel, 1, stride=stride, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.residual_unit(x)
        output += self.shortcut_unit(x)
        output = self.relu(output)

        return output


@deploy_efficient_layer.register()
class AnisotropicConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, p=0.2, stride=1, is_identify=True):
        """Anisotropic convolution block, including two layer convolution,
         instance normalization, drop out and ReLU"""
        super(AnisotropicConvBlock, self).__init__()
        self.residual_unit = nn.Sequential(
            ConvINReLU3D(in_channel, out_channel, kernel_size=(3, 3, 1), stride=stride, padding=(1, 1, 0), p=p),
            ConvIN3D(out_channel, out_channel, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1)))
        self.shortcut_unit = nn.Sequential() if stride == 1 and in_channel == out_channel and is_identify else \
            ConvIN3D(in_channel, out_channel, kernel_size=1, stride=stride, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.residual_unit(x)
        output += self.shortcut_unit(x)
        output = self.relu(output)

        return output


@NETWORK_REGISTRY.register()
class DeployEfficientSegNet(nn.Module):

    def __init__(self, cfg=None):
        super().__init__()

        # EfficientSegNet parameter.
        num_class = cfg.num_class
        num_channel = cfg.num_channel
        num_blocks = cfg.num_blocks
        decoder_num_block = cfg.decoder_num_block
        self.num_depth = cfg.num_depth
        self.is_preprocess = cfg.is_preprocess
        self.is_postprocess = cfg.is_postprocess

        encoder_conv_block = deploy_efficient_layer.get(cfg.encoder_conv_block)
        decoder_conv_block = deploy_efficient_layer.get(cfg.decoder_conv_block)
        context_block = deploy_efficient_layer.get(cfg.context_block)

        self.input = InputLayer(input_size=cfg.input_size, clip_window=cfg.clip_window)
        self.output = OutputLayer()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv0_0 = self._mask_layer(encoder_conv_block, 1, num_channel[0], num_blocks[0], stride=1)
        self.conv1_0 = self._mask_layer(encoder_conv_block, num_channel[0], num_channel[1], num_blocks[0], stride=2)
        self.conv2_0 = self._mask_layer(encoder_conv_block, num_channel[1], num_channel[2], num_blocks[1], stride=2)
        self.conv3_0 = self._mask_layer(encoder_conv_block, num_channel[2], num_channel[3], num_blocks[2], stride=2)
        self.conv4_0 = self._mask_layer(encoder_conv_block, num_channel[3], num_channel[4], num_blocks[3], stride=2)

        if context_block is not None:
            context_kernel_size = [i // 16 for i in cfg.input_size]
            self.context_block = context_block(num_channel[4], num_channel[4], kernel_size=context_kernel_size)
        else:
            self.context_block = nn.Sequential()

        self.trans_4 = ConvINReLU3D(num_channel[4], num_channel[3], kernel_size=1, stride=1, padding=0, p=0.2)
        self.trans_3 = ConvINReLU3D(num_channel[3], num_channel[2], kernel_size=1, stride=1, padding=0, p=0.2)
        self.trans_2 = ConvINReLU3D(num_channel[2], num_channel[1], kernel_size=1, stride=1, padding=0, p=0.2)
        self.trans_1 = ConvINReLU3D(num_channel[1], num_channel[0], kernel_size=1, stride=1, padding=0, p=0.2)

        self.conv3_1 = self._mask_layer(decoder_conv_block, num_channel[3],
                                        num_channel[3], decoder_num_block, stride=1)
        self.conv2_2 = self._mask_layer(decoder_conv_block, num_channel[2],
                                        num_channel[2], decoder_num_block, stride=1)
        self.conv1_3 = self._mask_layer(decoder_conv_block, num_channel[1],
                                        num_channel[1], decoder_num_block, stride=1)
        self.conv0_4 = self._mask_layer(decoder_conv_block, num_channel[0],
                                        num_channel[0], decoder_num_block, stride=1)

        self.final = nn.Conv3d(num_channel[0], num_class, kernel_size=1, bias=False)

        self._initialize_weights()

    def _mask_layer(self, block, in_channels, out_channels, num_block, stride):
        layers = []
        layers.append(block(in_channels, out_channels, p=0.2, stride=stride, is_identify=False))
        for _ in range(num_block-1):
            layers.append(block(out_channels, out_channels, p=0.2, stride=1, is_identify=True))

        return nn.Sequential(*layers)

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
        x1_0 = self.conv1_0(x)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)
        x4_0 = self.conv4_0(x3_0)
        x4_0 = self.context_block(x4_0)

        x3_0 = self.conv3_1(self.up(self.trans_4(x4_0)) + x3_0)
        x2_0 = self.conv2_2(self.up(self.trans_3(x3_0)) + x2_0)
        x1_0 = self.conv1_3(self.up(self.trans_2(x2_0)) + x1_0)
        x = self.conv0_4(self.up(self.trans_1(x1_0)) + x)

        x = self.final(x)
        if self.is_postprocess:
            x = self.output(x, x_input)
        x = F.sigmoid(x)

        return x

