import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


####################################### MODEL DEFINATION #########################################

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            # Replace with ReLU
            nn.ReLU(inplace=False)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup, momentum=0.1),
        ])
        self.conv = nn.Sequential(*layers)
        # Replace torch.add with floatfunctional
        # self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
            # return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        # if inverted_residual_setting is None:
        #     inverted_residual_setting = [
        #         # t, c, n, s
        #         [1, 16, 1, 1],
        #         [6, 24, 2, 2],
        #         [6, 32, 3, 2],
        #         [6, 64, 4, 2],
        #         [6, 96, 3, 1],
        #         [6, 160, 3, 2],
        #         [6, 320, 1, 1],
        #     ]
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [[1, 16, 1, 1],
                 [6, 24, 2, 2],
                 [6, 32, 3, 2]],
                [[6, 64, 4, 2],
                 [6, 96, 3, 1]],
                [[6, 160, 3, 2],
                 [6, 320, 1, 1]],
            ]

        # # only check the first element, assuming user knows t,c,n,s are required
        # if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
        #     raise ValueError("inverted_residual_setting should be non-empty "
        #                      "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        # print('first_stage-input_channel:', input_channel)
        # self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        # print('last_stage-output_channel:', self.last_channel)

        self.head = ConvBNReLU(3, input_channel, stride=2)
        # building inverted residual blocks
        self.stages = nn.ModuleList()
        for stage_setting in inverted_residual_setting:
            layers = []
            for t, c, n, s in stage_setting:
                output_channel = _make_divisible(c * width_mult, round_nearest)
                for i in range(n):
                    stride = s if i == 0 else 1
                    layers.append(block(input_channel, output_channel, stride, expand_ratio=t))
                    input_channel = output_channel
            self.stages.append(nn.Sequential(*layers))
        # building last several layers
        # self.tail = ConvBNReLU(input_channel, self.last_channel, kernel_size=1)

        # building classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(self.last_channel, num_classes),
        # )

        # weight initialization
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

    def forward(self, x):
        x = self.head(x)
        features = []
        for m in self.stages:
            x = m(x)
            features.append(x)
        return features




class HopenetMBV2(nn.Module):
    # Hopenet with 3 output layers for yaw, pitch and roll
    # Predicts Euler angles by binning and regression with the expected value
    def __init__(self, num_bins=66):
        self.inplanes = 64
        super(HopenetMBV2, self).__init__()
        self.backbone = MobileNetV2()
        self.avgpool = nn.AvgPool2d(4)
        self.fc_yaw = nn.Linear(320, num_bins)
        self.fc_pitch = nn.Linear(320, num_bins)
        self.fc_roll = nn.Linear(320, num_bins)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        if not self.training:
            # output = torch.cat((pre_yaw, pre_pitch, pre_roll), dim=0)
            # output = F.softmax(output, dim=1)
            output = torch.cat((pre_yaw[:, None, :], pre_pitch[:, None, :], pre_roll[:, None, :]), dim=1)
            output = F.softmax(output, dim=-1)#[b,3,bins]
            return output
        return pre_yaw, pre_pitch, pre_roll

###############################################################################################


class HeadPoseEstAPI(object):

    def __init__(self,
                 model_path='/data/output/head_pose_estimate_hopenet_mbv2_biwi_v3/model_122.pth'):
        super(HeadPoseEstAPI, self).__init__()
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        model = HopenetMBV2(num_bins=60)
        save_dict = torch.load(model_path, map_location='cpu')
        save_dict = save_dict['state_dict'] if 'state_dict' in save_dict.keys() else save_dict
        save_dict = {key.replace('module.', '') if key.startswith('module.') else key: val
                     for key, val in save_dict.items()}
        model.load_state_dict(save_dict)
        model.eval()
        return model

    def _img_preprocess(self, img_cv2):
        img_cv2 = cv2.resize(img_cv2, (112, 112))
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img = np.asarray(img_cv2, np.float32)
        rgb_mean = np.array([123.675, 116.28, 103.53], np.float32)
        rgb_std = np.array([58.395, 57.12, 57.375], np.float32)
        img -= rgb_mean
        img /= rgb_std
        img = np.transpose(img, axes=(2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img_t = torch.from_numpy(img)
        return img_t

    def __call__(self, img_cv2):
        img_t = self._img_preprocess(img_cv2)
        output = self.model(img_t)
        output = output.cpu().detach().numpy()[0]#[3,bins]
        print(output.shape)

        # decode
        idx_t = np.arange(60)[np.newaxis, :]
        angles = np.sum(output * idx_t, axis=1)*3 - 90
        yaw, pitch, roll = list(angles)

        if yaw >= -15 and yaw <= 25 and pitch >= -32 and pitch <= 22 and roll >= -23 and roll <= 23:
            is_valid = True
        else:
            is_valid = False

        return dict(is_valid=is_valid, metrics=dict(yaw=yaw, pitch=pitch, roll=roll))



def test():

    hpe_api = HeadPoseEstAPI()

    img_cv2 = cv2.imread('../datasets/test_crop.png', cv2.IMREAD_COLOR)
    # img_cv2 = cv2.imread('/data/FaceRecog/deploy/0.jpg')
    print('org_img:', img_cv2.shape)
    result = hpe_api(img_cv2)
    print(result)
    # yaw, pitch, roll = result['metrics']['yaw'], result['metrics']['pitch'], result['metrics']['roll']
    # from utils import visual
    # draw_img = visual.draw_axis(img_cv2.copy(), yaw, pitch, roll)
    # cv2.imwrite('result3_crop.jpg', draw_img)


if __name__ == '__main__':

    test()
