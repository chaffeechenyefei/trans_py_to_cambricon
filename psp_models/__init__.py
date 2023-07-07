from .psphead import PSPHead
from .fcn_head import  FCNHead
from .resnet import ResNet
from .convmodule import ConvModule
import torch.nn as nn
__all__ =["PSPHead","FCNHead","ResNet","PspNet","ConvModule"]


class PspNet(nn.Module):
	def __init__(self,cfg):
		super(PspNet,self).__init__()
		self.backbone = ResNet(**cfg["backbone"])
		self.decode_head = PSPHead(**cfg["decode_head"])
		# self.auxiliary_head = FCNHead(**cfg["auxiliary_head"])
	def forward(self,input):
		out = self.backbone(input)
		out = self.decode_head(out)
		# out = self.auxiliary_head(out)
		return out

if __name__=="__main__":
	model_config = dict(
		backbone=dict(
			depth=18,
			num_stages=4,
			out_indices=(0, 1, 2, 3),
			dilations=(1, 1, 2, 4),
			strides=(1, 2, 1, 1),
			deep_stem=True,
			avg_down=False,
			norm_eval=False,
			contract_dilation=True),
		decode_head=dict(
			in_channels=512,
			in_index=3,
			channels=128,
			pool_scales=(1, 2, 4, 8),
			dropout_ratio=0.1,
			num_classes=2,
			align_corners=False),
		auxiliary_head=dict(
			in_channels=256,
			in_index=2,
			channels=64,
			num_convs=1,
			concat_input=False,
			dropout_ratio=0.1,
			num_classes=2,
			align_corners=False))

	build_pspnet(model_config)