# Copyright (c) OpenMMLab. All rights reserved.
# from mmcv.cnn import build_conv_layer, build_norm_layer
from torch import nn as nn
# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from .convmodule import ConvModule


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels,act_cfg, align_corners, **kwargs):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        act_cfg=self.act_cfg,
                        **kwargs)))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out,
                size=x.size()[2:],
                scale_factor=None,
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class PSPHead(nn.Module):
    """Pyramid Scene Parsing Network.

    This head is the implementation of
    `PSPNet <https://arxiv.org/abs/1612.01105>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
    """
    def __init__(self,in_channels,channels,num_classes, pool_scales=(1, 2, 3, 6), dropout_ratio=0.1,\
                 act_cfg = dict(type="ReLU"),in_index=-1,input_transform=None,align_corners=False):
        super(PSPHead, self).__init__()
        assert isinstance(pool_scales, (list, tuple))
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.align_corners = align_corners
        self.pool_scales = pool_scales
        self.psp_modules = PPM(
            self.pool_scales,
            self.in_channels,
            self.channels,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            act_cfg=self.act_cfg)

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        #chaffee
        self.softmax = nn.Softmax(dim=1)
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

		The in_channels, in_index and input_transform must match.
		Specifically, when input_transform is None, only single feature map
		will be selected. So in_channels and in_index must be of type int.
		When input_transform

		Args:
			in_channels (int|Sequence[int]): Input channels.
			in_index (int|Sequence[int]): Input feature index.
			input_transform (str|None): Transformation type of input features.
				Options: 'resize_concat', 'multiple_select', None.
				'resize_concat': Multiple feature maps will be resize to the
					same size as first one and than concat together.
					Usually used in FCN head of HRNet.
				'multiple_select': Multiple feature maps will be bundle into
					a list and passed into decode head.
				None: Only one select feature map is allowed.
		"""

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels


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
        x = self._transform_inputs(inputs)###根据index选择固定stage的out feature 进行psp
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        output = self.cls_seg(output)
        output = self.softmax(output)
        return output


if __name__=="__main__":
    in_channels = 512
    in_index = 3
    channels = 128
    pool_scales = (1, 2, 4, 8)
    dropout_ratio = 0.1
    num_classes = 2
    align_corners = False

    backbone = PSPHead(in_channels,channels,2,pool_scales,dropout_ratio)
    print(backbone)