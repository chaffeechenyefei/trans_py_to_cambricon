import torch
import torch.nn as nn
import cv2
import numpy as np
from faceattr_models.efficientnet.model import EfficientNet


class ProxyConvModule(nn.Module):
    """
    groups!=1 because ERR::CNML conv2d_first only support groups =1, but got 3
    """
    def __init__(self):
        super(ProxyConvModule, self).__init__()
        self.kernel_size = (3,3)
        self.in_channels = 3
        self.out_channels = 3
        self.stride = 1
        self.padding = 1
        self.groups = 1
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride,
                              self.padding, groups=self.groups,bias=False)

        a = torch.zeros(1, 1, 3, 3)
        a[:,:,1,1] = 1
        b = torch.zeros(1, 1, 3, 3)
        k1 = torch.cat([a, b, b], dim=1)
        k2 = torch.cat([b, a, b], dim=1)
        k3 = torch.cat([b, b, a], dim=1)
        k = torch.cat([k1, k2, k3], dim=0)

        self.conv.weight = nn.Parameter(k.float())

    def forward(self, input):
        return self.conv(input)


class faceAttrModel_MLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.proxyconv = ProxyConvModule()
        self.model = EfficientNet.from_pretrained('efficientnet-b3', in_channels=3, load_fc=False, image_size=112,
                                             include_top=False)

        # self.age_weights = nn.Parameter( torch.arange(0, 101).reshape(1, 101).float() )

    def forward(self, input):
        """
        :param input:112x112 
        :return: 
        """
        input = self.proxyconv(input)
        gender_out, age_out = self.model(input)
        # age_out = torch.sum(self.age_weights * age_out, dim=1)
        # print(gender_out.shape, age_out.shape) torch.Size([b, 2]) torch.Size([b, 101])
        return torch.cat([gender_out, age_out], dim=1)

    def load_checkpoint(self, pthfile):
        save_dict = torch.load(pthfile, map_location="cpu")
        save_dict = save_dict['state_dict'] if 'state_dict' in save_dict.keys() else save_dict
        # """特殊处理, 针对多卡训练结果"""
        # save_dict = {k.replace('module.',''):v for k,v in save_dict.items()}
        self.model.load_state_dict(save_dict, strict=True)
        return self




"""
origin file
"""
class Inference(nn.Module):
    def __init__(self, model_fp, img_size=112, use_cuda=True, padding=True):
        self.use_cuda = use_cuda
        self.img_size = img_size
        self.padding = padding

        model = EfficientNet.from_pretrained('efficientnet-b3', in_channels=3, load_fc=False, image_size=112,
                                             include_top=False, weights_path=model_fp)
        self.model = model.cuda() if self.use_cuda else model
        self.model.eval()
        self.age_weights = torch.arange(0, 101).reshape(1, 101)
        self.age_weights = self.age_weights.cuda() if self.use_cuda else self.age_weights

    def img_padding(self, img_cv):
        h, w, c = img_cv.shape
        max_hw = max(h, w)
        top, left = int(0.5 * (max_hw - h)), int(0.5 * (max_hw - w))
        bottom, right = max_hw - h - top, max_hw - w - left
        img_cv = cv2.copyMakeBorder(img_cv, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        return img_cv

    def normal_imagenet(self, img_cv):
        mean = [123.675, 116.28, 103.53]
        std = [58.395, 57.12, 57.375]
        return (img_cv - mean) / std

    def hwc_to_chw(self, img_cv):
        return np.transpose(img_cv, axes=(2, 0, 1))

    def img_preprocess(self, img):
        if self.padding:
            img = self.img_padding(img)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.normal_imagenet(img)
        img = self.hwc_to_chw(img)

        data_t = torch.tensor(img, dtype=torch.float32)
        data_t = torch.unsqueeze(data_t, dim=0)
        data_t = data_t.cuda() if self.use_cuda else data_t
        return data_t

    def infer(self, img):
        data_t = self.img_preprocess(img)

        gender_out, age_out = self.model(data_t)
        gender_out = gender_out.cpu().data.numpy()

        age_out = torch.sum(self.age_weights * age_out, dim=1)
        age_out = age_out.cpu().data.numpy()

        return gender_out, age_out

    def infer_imdb(self, img, label_fp):
        H, W, _ = img.shape
        face_info = np.loadtxt(label_fp).astype(int)
        x0, y0, x1, y1, score = face_info[:5]
        w, h = x1 - x0, y1 - y0
        x0, y0 = max(0, x0 - w / 2), max(0, y0 - h / 2)
        x1, y1 = min(W, x1 + w / 2), min(H, y1 + h / 2)
        img = img[int(y0):int(y1), int(x0):int(x1), :]

        return self.infer(img)