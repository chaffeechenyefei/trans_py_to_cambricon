import cv2
import numpy as np
import torch
import torch.nn as nn

def normal_imagenet(img_cv):
    """
    :param img_cv: [h,w,c] bgr
    :return:
    """
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    if isinstance(img_cv,list):
        return [ (im -mean)/std for im in img_cv ]
    else:
        return (img_cv-mean)/std

def hwc_to_chw(img_cv):
    if isinstance(img_cv,list):
        return [  np.transpose(im,axes=(2,0,1)) for im in img_cv ]
    else:
        return np.transpose(img_cv,axes=(2,0,1))

def img_preprocess(imgs, dst_size=(736, 416), use_cuda=True):
    imgs = cv2.absdiff(imgs[0], imgs[1])
    imgs = cv2.resize(imgs, dst_size)
    imgs = normal_imagenet(imgs)
    imgs = hwc_to_chw(imgs)

    imgs = torch.FloatTensor(imgs.astype(np.float32))
    imgs = imgs.unsqueeze(dim=0)  # [1,c,h,w]
    if use_cuda:
        imgs = imgs.cuda()
    return imgs

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class DIFFUNet4(nn.Module):
    """
    PWC-DC net. add dilation convolution and densenet connections

    """

    def __init__(self, in_ch=3, out_ch=1):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping

        """
        super(DIFFUNet4, self).__init__()
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x.shape = (b, 3, h, w)

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        d4 = self.Up4(e4)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out

class Inference(object):
    def __init__(self, model=None, ckpt_path=None, use_cuda=True):
        super(Inference,self).__init__()
        self.use_cuda = use_cuda
        self.model = model
        self.last_layer = nn.Sigmoid()
        if ckpt_path is not None:
            save_dict = torch.load(ckpt_path)
            save_dict = save_dict['state_dict'] if 'state_dict' in save_dict.keys() else save_dict
            self.model.load_state_dict(save_dict, strict=False)
        if use_cuda:
            self.model = self.model.cuda()
            self.last_layer = self.last_layer.cuda()
        self.model.eval()

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
        mask = self.batch_inference(img) #numpy [1,h,w]
        mask = mask.squeeze()
        return mask


class BlurModule(nn.Module):
    """
    CNML conv2d_first only support groups =1, but got 3
    """
    def __init__(self):
        super(BlurModule, self).__init__()
        self.kernel_size = (3,3)
        self.in_channels = 3
        self.out_channels = 3
        self.stride = 1
        self.padding = 1
        self.groups = 1
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride,
                              self.padding, groups=self.groups,bias=False)

        a = torch.ones(1, 1, 3, 3) / (3 * 3)
        b = torch.zeros(1, 1, 3, 3)
        k1 = torch.cat([a, b, b], dim=1)
        k2 = torch.cat([b, a, b], dim=1)
        k3 = torch.cat([b, b, a], dim=1)
        k = torch.cat([k1, k2, k3], dim=0)

        self.conv.weight = nn.Parameter(k.float())

    def forward(self, input):
        return self.conv(input)

# class NormModule(nn.Module):
#     def __init__(self):
#         super(NormModule, self).__init__()
#         self.mean = nn.Parameter(   torch.tensor([123.675/255, 116.28/255, 103.53/255]).float() )
#         self.std = nn.Parameter(    torch.tensor([255/58.395, 255/57.12, 255/57.375]).float() )
#
#     def forward(self, input):
#         """
#         :param input: [b,c,h,w]
#         :return: [b,c,h,w]
#         """
#         # x = input/255. - self.mean.view(3,1,1)
#         x = input*(self.std.view(3,1,1))
#         return x



class DIFFUNet4_MLU(nn.Module):
    """
    DIFFUNet4 for MLU device
    input: [2b,c,h,w]
    output: [b,1,h,w]
    """

    def __init__(self, in_ch=3, out_ch=1):
        """
        """
        super(DIFFUNet4_MLU, self).__init__()
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]

        self.conv_proxy = BlurModule()

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        """
        t = 2
        :param input: [bt,c,h,w]
        :return: [b,1,h,w]
        """
        t = 2
        # print("1")
        # [bt,c,h,w] -> [bt, c, h ,w]
        x1 = self.conv_proxy(input)
        # print("2")
        x11 = ( x1[:,0,...] - 123.675 )/58.395
        x12 = ( x1[:,1,...] - 116.28 )/57.12
        x13 = ( x1[:,2,...] - 103.53 )/57.375
        x2 = torch.stack([x11,x12,x13],dim=1)

        # x2 = self.norm(x1)
        # x2 = x1*1
        # print("3")
        # [bt, c , h, w] -> [b,t,c,h,w]
        x3 = x2.reshape(tuple([-1, t, x2.shape[1]]) + x2.shape[2:])
        # [b,t,c,h,w] -> [b,c,h,w]
        # print("4")
        x4 = torch.abs( x3[:,0,...] - x3[:,1,...] )
        # x4 = torch.abs(torch.mean(x3,dim=1))
        # print("5")

        e1 = self.Conv1(x4)
        # print("6")

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        # print("7")
        d4 = self.Up4(e4)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        # print("8")

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        # print("9")

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        # print("10")
        out = self.Conv(d2)
        # print("11")
        out = self.sigmoid(out)
        # print("12")

        return out