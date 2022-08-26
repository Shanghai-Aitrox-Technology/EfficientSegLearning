
import torch
import torch.nn as nn

from BaseSeg.models.blocks.residual_block import ResFourLayerConvBlock, ResTwoLayerConvBlock
from BaseSeg.models.blocks.process_block import InputLayer, OutputLayer
from BaseSeg.models.blocks.convNeXt_block import StemLayer, ConvNeXtBlock

from .builder import NETWORK_REGISTRY


@NETWORK_REGISTRY.register()
class ConvNeXtSegNet(nn.Module):

    def __init__(self, cfg=None):
        super().__init__()

        # UNet parameter.
        num_class = cfg.num_class
        num_channel = cfg.num_channel
        num_blocks = cfg.num_blocks
        self.is_preprocess = cfg.is_preprocess
        self.is_postprocess = cfg.is_postprocess
        self.is_dynamic_empty_cache = cfg.is_dynamic_empty_cache

        self.input = InputLayer(input_size=cfg.input_size, clip_window=cfg.clip_window)
        self.output = OutputLayer()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv0_0 = StemLayer(1, num_channel[0])
        self.conv1_0 = ConvNeXtBlock(num_channel[0], num_channel[1],
                                     down_sample=2, num_residual=num_blocks[0])
        self.conv2_0 = ConvNeXtBlock(num_channel[1], num_channel[2],
                                     down_sample=2, num_residual=num_blocks[1])
        self.conv3_0 = ConvNeXtBlock(num_channel[2], num_channel[3],
                                     down_sample=2, num_residual=num_blocks[2])
        self.conv4_0 = ConvNeXtBlock(num_channel[3], num_channel[4],
                                     down_sample=2, num_residual=num_blocks[3])

        self.conv3_1 = ResTwoLayerConvBlock(num_channel[3] + num_channel[4], num_channel[3], num_channel[3],
                                            is_dynamic_empty_cache=self.is_dynamic_empty_cache)
        self.conv2_2 = ResTwoLayerConvBlock(num_channel[2] + num_channel[3], num_channel[2], num_channel[2],
                                            is_dynamic_empty_cache=self.is_dynamic_empty_cache)
        self.conv1_3 = ResTwoLayerConvBlock(num_channel[1] + num_channel[2], num_channel[1], num_channel[1],
                                            is_dynamic_empty_cache=self.is_dynamic_empty_cache)
        self.conv0_4 = ResTwoLayerConvBlock(num_channel[0] + num_channel[1], num_channel[0], num_channel[0],
                                            is_dynamic_empty_cache=self.is_dynamic_empty_cache)

        self.final = nn.Conv3d(num_channel[0], num_class, kernel_size=(1, 1, 1), bias=False)

        self._initialize_weights()
        # self.final.bias.data.fill_(-2.19)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out_size = x.shape[2:]
        if self.is_preprocess:
            x = self.input(x)

        x = self.conv0_0(x)
        x1_0 = self.conv1_0(x)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)
        x4_0 = self.conv4_0(x3_0)

        x3_0 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        if self.is_dynamic_empty_cache:
            del x4_0
            torch.cuda.empty_cache()

        x2_0 = self.conv2_2(torch.cat([x2_0, self.up(x3_0)], 1))
        if self.is_dynamic_empty_cache:
            del x3_0
            torch.cuda.empty_cache()

        x1_0 = self.conv1_3(torch.cat([x1_0, self.up(x2_0)], 1))
        if self.is_dynamic_empty_cache:
            del x2_0
            torch.cuda.empty_cache()

        x = self.conv0_4(torch.cat([x, self.up(x1_0)], 1))
        if self.is_dynamic_empty_cache:
            del x1_0
            torch.cuda.empty_cache()

        x = self.final(x)
        if self.is_postprocess:
            x = self.output(x, out_size)

        return x
