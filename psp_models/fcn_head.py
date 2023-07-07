# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from .convmodule import  ConvModule

class FCNHead(nn.Module):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 dropout_ratio=0.1,
                 act_cfg = dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 align_corners=None):
        super(FCNHead, self).__init__()
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.in_channels = in_channels
        self.channels  = channels
        self.num_classes = num_classes
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.input_transform = input_transform
        self.align_corners = align_corners
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.dropout_ratio = dropout_ratio

        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                act_cfg=self.act_cfg)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                F.interpolate(
                    input=x,
                    size=inputs[0].shape[2:],
                    scale_factor=None,
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output


if __name__=="__main__":
    in_channels = 256
    in_index = 2
    channels = 64
    num_convs = 1
    concat_input = False
    dropout_ratio = 0.1
    num_classes = 2
    align_corners = False

    backbone = FCNHead(in_channels,channels,num_classes=num_classes,num_convs=1, dropout_ratio=dropout_ratio,concat_input=concat_input)
    print(backbone)

