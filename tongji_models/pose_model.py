import torch
import torch.nn as nn
import torch.nn.functional as F


class FasePoseR50_MLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FastPose('resnet50', 17)

    def forward(self, x):
        return self.model(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        return self


    # def load_checkpoint(self, pthfile):
    #     save_dict = torch.load(pthfile, map_location="cpu")
    #     save_dict = save_dict['state_dict'] if 'state_dict' in save_dict.keys() else save_dict
    #     # """特殊处理, 针对多卡训练结果"""
    #     # save_dict = {k.replace('module.',''):v for k,v in save_dict.items()}
    #     self.model.load_state_dict(save_dict, strict=True)
    #     return self


class myPixelShuffle(nn.Module):
    """
    https://blog.csdn.net/ONE_SIX_MIX/article/details/103757856
    """
    def __init__(self, upscale_factor):
        super().__init__()
        self.r = upscale_factor

    def forward(self, x):
        b, c, h, w = x.shape
        r = self.r
        # output size
        _c, _h, _w = [c // (r * r), h * r, w * r]
        x = x.reshape(b, _c, r, r, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3)  # [b,_c,h,r,w,r]
        x = x.reshape(b, _c, _h, _w)
        return x


class SPPE_FastPose(object):
    def __init__(self,
                 backbone,
                 input_height=224,
                 input_width=192,
                 device='cuda'):
        assert backbone in ['resnet50', 'resnet101'], '{} backbone is not support yet!'.format(backbone)

        self.inp_h = input_height
        self.inp_w = input_width
        self.device = device

        self.model = InferenNet_fastRes50().to(device)
        self.model.eval()


    # def predict(self, image, bboxs, bboxs_scores, ignore=0.005):
    #     inps, pt1, pt2 = crop_dets(image, bboxs, self.inp_h, self.inp_w, ignore=ignore)
    #     pose_hm = self.model(inps.to(self.device)).cpu().data
    #
    #     # Cut eyes and ears.
    #     pose_hm = torch.cat([pose_hm[:, :1, ...], pose_hm[:, 5:, ...]], dim=1)  # 删除掉[1,2,3,4]左眼、右眼、左耳、右耳共4个点 @yjy
    #
    #     xy_hm, xy_img, scores = getPrediction(pose_hm, pt1, pt2, self.inp_h, self.inp_w,
    #                                           pose_hm.shape[-2], pose_hm.shape[-1])
    #     result = pose_nms(bboxs, bboxs_scores, xy_img, scores)
    #     return result


class InferenNet_fastRes50(nn.Module):
    def __init__(self, weights_file='./Models/sppe/fast_res50_256x192.pth'):
        super().__init__()

        self.pyranet = FastPose('resnet50', 17).cuda()
        print('Loading pose model from {}'.format(weights_file))
        self.pyranet.load_state_dict(torch.load(weights_file))
        self.pyranet.eval()

    def forward(self, x):
        out = self.pyranet(x)

        return out


class FastPose(nn.Module):
    DIM = 128

    def __init__(self, backbone='resnet101', num_join=17):
        super(FastPose, self).__init__()
        assert backbone in ['resnet50', 'resnet101']

        self.preact = SEResnet(backbone)

        # self.shuffle1 = nn.PixelShuffle(2)
        self.shuffle1 = myPixelShuffle(2)
        self.duc1 = DUC(512, 1024, upscale_factor=2)
        self.duc2 = DUC(256, 512, upscale_factor=2)

        self.conv_out = nn.Conv2d(
            self.DIM, num_join, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.preact(x)
        out = self.shuffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)

        out = self.conv_out(out)
        return out


class DUC(nn.Module):
    """
    INPUT: inplanes, planes, upscale_factor
    OUTPUT: (planes // 4)* ht * wd
    """
    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        # self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.pixel_shuffle = myPixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


class SEResnet(nn.Module):
    """ SEResnet """

    def __init__(self, architecture):
        super(SEResnet, self).__init__()
        assert architecture in ["resnet50", "resnet101"]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.inplanes = 64
        self.layers = [3, 4, {"resnet50": 6, "resnet101": 23}[architecture], 3]

        self.block = Bottleneck

        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(self.block, 64, self.layers[0])
        self.layer2 = self.make_layer(
            self.block, 128, self.layers[1], stride=2)
        self.layer3 = self.make_layer(
            self.block, 256, self.layers[2], stride=2)

        self.layer4 = self.make_layer(
            self.block, 512, self.layers[3], stride=2)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # 64 * h/4 * w/4
        x = self.layer1(x)  # 256 * h/4 * w/4
        x = self.layer2(x)  # 512 * h/8 * w/8
        x = self.layer3(x)  # 1024 * h/16 * w/16
        x = self.layer4(x)  # 2048 * h/32 * w/32
        return x

    # def stages(self):
    #     return [self.layer1, self.layer2, self.layer3, self.layer4]

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if downsample is not None:
            layers.append(block(self.inplanes, planes, stride, downsample, reduction=True))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        if reduction:
            self.se = SELayer(planes * 4)

        self.reduc = reduction
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.reduc:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y