import os
import time
import numpy as np
import torch
import torch.nn as nn
import cv2

from dataloader.normalize import hwc_to_chw,normal_imagenet

from models.unet import UNet3D, UNet2DShift, UNetResNet18_Shift, UNetResNet18, UNetResNet18_BN
from checkpoint.checkpoint import CheckpointMgr
pj = os.path.join

class unetResNet18_BN_Inference(object):
    def __init__(self, ckpt_path=None, use_cuda=True):
        super(unetResNet18_BN_Inference,self).__init__()
        self.use_cuda = use_cuda
        self.model = UNetResNet18_BN(n_classes=1,n_segment=2, use_bn=False)
        self.model = self.model.initial()
        self.last_layer = nn.Sigmoid()
        if ckpt_path is not None:
            self.load(ckpt_path)
        if use_cuda:
            self.model = self.model.cuda()
            self.last_layer = self.last_layer.cuda()
        self.model.eval()

    def load(self, ckpt_path):
        ckpt_op = CheckpointMgr(ckpt_dir=ckpt_path)
        ckpt_op.load_checkpoint(self.model,map_location='cpu')
        return self

    @torch.no_grad()
    def batch_inference(self, img):
        """
        :param img: [b,c,t,h,w] 
        :return: [b,h,w] numpy 
        """
        return (self.last_layer(self.model(img))).data.cpu().numpy()

    def inference(self, img):
        """
        :param img: [ [h,w,c] ], bgr, list of numpy
        :return: [1,h,w] numpy 
        """
        img = normal_imagenet(img)
        img = hwc_to_chw(img)
        img = [torch.FloatTensor(im.astype(np.float32)) for im in img]  # [ [c,h,w] ]
        img = torch.stack(img, dim=1)  # [c,t,h,w]
        img = img.unsqueeze(dim=0) #[1,c,t,h,w]
        if self.use_cuda:
            img = img.cuda()
        mask = self.batch_inference(img) #numpy [1,h,w]
        mask = mask.squeeze()
        return mask

class unetResNet18_Inference(object):
    def __init__(self, ckpt_path=None, use_cuda=True):
        super(unetResNet18_Inference,self).__init__()
        self.use_cuda = use_cuda
        self.model = UNetResNet18(n_classes=1,n_segment=2)
        self.model = self.model.initial()
        self.last_layer = nn.Sigmoid()
        if ckpt_path is not None:
            self.load(ckpt_path)
        if use_cuda:
            self.model = self.model.cuda()
            self.last_layer = self.last_layer.cuda()
        self.model.eval()

    def load(self, ckpt_path):
        ckpt_op = CheckpointMgr(ckpt_dir=ckpt_path)
        ckpt_op.load_checkpoint(self.model,map_location='cpu')
        return self

    @torch.no_grad()
    def batch_inference(self, img):
        """
        :param img: [b,c,t,h,w] 
        :return: [b,h,w] numpy 
        """
        return (self.last_layer(self.model(img))).data.cpu().numpy()

    def inference(self, img):
        """
        :param img: [ [h,w,c] ], bgr, list of numpy
        :return: [1,h,w] numpy 
        """
        img = normal_imagenet(img)
        img = hwc_to_chw(img)
        img = [torch.FloatTensor(im.astype(np.float32)) for im in img]  # [ [c,h,w] ]
        img = torch.stack(img, dim=1)  # [c,t,h,w]
        img = img.unsqueeze(dim=0) #[1,c,t,h,w]
        if self.use_cuda:
            img = img.cuda()
        mask = self.batch_inference(img) #numpy [1,h,w]
        mask = mask.squeeze()
        return mask

class unetResNet18TSM_Inference(object):
    def __init__(self, ckpt_path=None, use_cuda=True):
        super(unetResNet18TSM_Inference,self).__init__()
        self.use_cuda = use_cuda
        self.model = UNetResNet18_Shift(n_classes=1,n_segment=2)
        self.model = self.model.initial()
        self.last_layer = nn.Sigmoid()
        if ckpt_path is not None:
            self.load(ckpt_path)
        if use_cuda:
            self.model = self.model.cuda()
            self.last_layer = self.last_layer.cuda()
        self.model.eval()

    def load(self, ckpt_path):
        ckpt_op = CheckpointMgr(ckpt_dir=ckpt_path)
        ckpt_op.load_checkpoint(self.model,map_location='cpu')
        return self

    @torch.no_grad()
    def batch_inference(self, img):
        """
        :param img: [b,c,t,h,w] 
        :return: [b,h,w] numpy 
        """
        return (self.last_layer(self.model(img))).data.cpu().numpy()

    def inference(self, img):
        """
        :param img: [ [h,w,c] ], bgr, list of numpy
        :return: [1,h,w] numpy 
        """
        img = normal_imagenet(img)
        img = hwc_to_chw(img)
        img = [torch.FloatTensor(im.astype(np.float32)) for im in img]  # [ [c,h,w] ]
        img = torch.stack(img, dim=1)  # [c,t,h,w]
        img = img.unsqueeze(dim=0) #[1,c,t,h,w]
        if self.use_cuda:
            img = img.cuda()
        mask = self.batch_inference(img) #numpy [1,h,w]
        mask = mask.squeeze()
        return mask

class unet2dshift_Inference(object):
    def __init__(self, ckpt_path=None, use_cuda=True):
        super(unet2dshift_Inference,self).__init__()
        self.use_cuda = use_cuda
        self.model = UNet2DShift(n_channels=3,n_classes=1,n_segment=3)
        self.model = self.model.initial()
        self.last_layer = nn.Sigmoid()
        if ckpt_path is not None:
            self.load(ckpt_path)
        if use_cuda:
            self.model = self.model.cuda()
            self.last_layer = self.last_layer.cuda()
        self.model.eval()

    def load(self, ckpt_path):
        ckpt_op = CheckpointMgr(ckpt_dir=ckpt_path)
        ckpt_op.load_checkpoint(self.model,map_location='cpu')
        return self

    @torch.no_grad()
    def batch_inference(self, img):
        """
        :param img: [b,c,t,h,w] 
        :return: [b,h,w] numpy 
        """
        return (self.last_layer(self.model(img))).data.cpu().numpy()

    def inference(self, img):
        """
        :param img: [ [h,w,c] ], bgr, list of numpy
        :return: [1,h,w] numpy 
        """
        img = normal_imagenet(img)
        img = hwc_to_chw(img)
        img = [torch.FloatTensor(im.astype(np.float32)) for im in img]  # [ [c,h,w] ]
        img = torch.stack(img, dim=1)  # [c,t,h,w]
        img = img.unsqueeze(dim=0) #[1,c,t,h,w]
        if self.use_cuda:
            img = img.cuda()
        mask = self.batch_inference(img) #numpy [1,h,w]
        mask = mask.squeeze()
        return mask




class unet3d_Inference(object):
    def __init__(self, ckpt_path=None, use_cuda=True):
        super(unet3d_Inference,self).__init__()
        self.use_cuda = use_cuda
        self.model = UNet3D(n_channels=3,n_classes=1,bilinear=True)
        self.model = self.model.initial()
        if ckpt_path is not None:
            self.load(ckpt_path)
        if use_cuda:
            self.model = self.model.cuda()
        self.model.eval()

    def load(self, ckpt_path):
        ckpt_op = CheckpointMgr(ckpt_dir=ckpt_path)
        ckpt_op.load_checkpoint(self.model,map_location='cpu')
        return self

    @torch.no_grad()
    def batch_inference(self, img):
        """
        :param img: [b,c,t,h,w] 
        :return: [b,h,w] numpy 
        """
        return (self.model(img)).data.cpu().numpy()

    def inference(self, img):
        """
        :param img: [ [h,w,c] ], bgr, list of numpy
        :return: [1,h,w] numpy 
        """
        img = normal_imagenet(img)
        img = hwc_to_chw(img)
        img = [torch.FloatTensor(im.astype(np.float32)) for im in img]  # [ [c,h,w] ]
        img = torch.stack(img, dim=1)  # [c,t,h,w]
        img = img.unsqueeze(dim=0) #[1,c,t,h,w]
        if self.use_cuda:
            img = img.cuda()
        mask = self.batch_inference(img) #numpy [1,h,w]
        mask = mask.squeeze()
        return mask

if __name__ == '__main__':
    T = 16
    net = unet3d_Inference( ckpt_path='./pth', use_cuda=True)
    img_path = '/project/data/google/tv04_mp4/frames/'
    img_list = [ c for c in os.listdir(img_path) if c.endswith('.bmp') ]
    img_list.sort()

    imgs = [ cv2.imread(pj(img_path,c)) for c in img_list[180:180+T] ]

    mask = net.inference(imgs)
    mask *= 255
    mask = mask.clip(0,255).astype(np.uint8)
    cv2.imwrite('./tmp.jpg',mask)