import torch
import torch.nn as nn
import torch.nn.functional as F

from smoking_models.resnet import resnet18, resnet34

model_dict = {'resnet18':resnet18, 'resnet34':resnet34}

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

class SmokingModel_1(nn.Module):
    def __init__(self, num_classes=2, model_name='resnet34'):
        super(SmokingModel_1, self).__init__(num_classes, model_name)
        if model_name not in model_dict:
            raise NotImplementedError('model:{} is not defined'.format(model_name))
        self.backbone_face = model_dict[model_name]() # n c w h
        self.backbone_hand = model_dict[model_name]() # n c w h
        self.fc = nn.Conv2d(512*3, num_classes, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _feat_weight(self, input):
        notzero_num = torch.sum(input!=0, (1,2,3))
        w_ = torch.sigmoid(notzero_num) # if notzero_num==0,input is zero matrix, w_=0.5;else w_=1;
        weight = 2*w_ - 1
        weight = weight.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # print(weight)
        return weight

    def forward(self, input):
        """
        :param input: [b,3,c,h,w] 
        :return: 
        """
        face_input = input[:, 0, :, :, :]
        face_feat = self.backbone_face(face_input)*self._feat_weight(face_input)
        hand1_input = input[:, 1, :, :, :]
        hand1_feat = self.backbone_hand(hand1_input)*self._feat_weight(hand1_input)
        hand2_input = input[:, 2, :, :, :]
        hand2_feat = self.backbone_hand(hand2_input)*self._feat_weight(hand2_input)

        feat = torch.cat([face_feat, hand1_feat, hand2_feat], dim=1)
        feat_pool = self.avgpool(feat)
        output = self.fc(feat_pool)
        output = output.squeeze()
        return output

class SmokingModel_1_MLU(nn.Module):
    """
    preprocess:
    bgr2rgb
    (x-mean)/std
    py_mean = [123.675, 116.28, 103.53]
    py_std = [58.395, 57.12, 57.375]
    """
    def __init__(self, num_classes=2, model_name='resnet34'):
        super(SmokingModel_1_MLU, self).__init__()
        self.proxy_first_conv = ProxyConvModule()
        if model_name not in model_dict:
            raise NotImplementedError('model:{} is not defined'.format(model_name))
        self.backbone_face = model_dict[model_name]() # n c w h
        self.backbone_hand = model_dict[model_name]() # n c w h
        self.fc = nn.Conv2d(512*3, num_classes, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_classes = num_classes

        self.null_input = nn.Parameter(torch.tensor([123.675/58.395, 116.28/57.12, 103.53/57.375]).reshape(1,3,1,1).float())

    def _feat_weight(self, input):
        # input = (raw - m)/s = raw/s - m/s
        # x = intput + m/s = raw/s
        x = input + self.null_input #
        notzero_num = torch.sum( x > 0.1 , (1,2,3)).float()
        w_ = torch.sigmoid(notzero_num) # if notzero_num==0,input is zero matrix, w_=0.5;else w_=1;
        weight = 2*w_ - 1
        weight = weight.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # print(weight)
        return weight

    def forward(self, input):
        """
        :param input: [b3,c,h,w]
        :return: [b,2] "non-smoking, smoking" ->[softmax, slice]-> "smoking" 
        """
        # [b3,c,h,w] -> [b3,c,h,w]
        x = self.proxy_first_conv(input)
        #[b3,c,h,w] -> [b,3,c,h,w]
        x = x.reshape( tuple([-1,3] + list(x.shape[1:])) )

        face_input = x[:, 0, :, :, :]
        face_feat = self.backbone_face(face_input)*self._feat_weight(face_input)
        hand1_input = x[:, 1, :, :, :]
        hand1_feat = self.backbone_hand(hand1_input)*self._feat_weight(hand1_input)
        hand2_input = x[:, 2, :, :, :]
        hand2_feat = self.backbone_hand(hand2_input)*self._feat_weight(hand2_input)

        feat = torch.cat([face_feat, hand1_feat, hand2_feat], dim=1)
        feat_pool = self.avgpool(feat)#[b,c,1,1]
        output = self.fc(feat_pool)#[b,c,1,1]
        output = output.reshape(-1, self.num_classes) #[b,c]
        output = F.softmax(output, dim=-1)
        # print(output.shape)
        output = output[:,1:2]
        # print(output.shape)
        return output



class SmokingModel_1_MIMO_MLU(nn.Module):
    """
    preprocess:
    bgr2rgb
    (x-mean)/std
    py_mean = [123.675, 116.28, 103.53]
    py_std = [58.395, 57.12, 57.375]
    """
    def __init__(self, num_classes=2, model_name='resnet34'):
        super(SmokingModel_1_MIMO_MLU, self).__init__()
        self.proxy_first_conv = ProxyConvModule()
        if model_name not in model_dict:
            raise NotImplementedError('model:{} is not defined'.format(model_name))
        self.backbone_face = model_dict[model_name]() # n c w h
        self.backbone_hand = model_dict[model_name]() # n c w h
        self.fc = nn.Conv2d(512*3, num_classes, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_classes = num_classes

        self.null_input = nn.Parameter(torch.tensor([123.675/58.395, 116.28/57.12, 103.53/57.375]).reshape(1,3,1,1).float())

    def _feat_weight(self, input):
        # input = (raw - m)/s = raw/s - m/s
        # x = intput + m/s = raw/s
        x = input + self.null_input #
        notzero_num = torch.sum( x > 0.1 , (1,2,3)).float()
        w_ = torch.sigmoid(notzero_num) # if notzero_num==0,input is zero matrix, w_=0.5;else w_=1;
        weight = 2*w_ - 1
        weight = weight.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # print(weight)
        return weight

    def forward(self, input0, input1, input2):
        """
        :param input: [b,c,h,w],[b,c,h,w],[b,c,h,w]
        :return: [b,2] "non-smoking, smoking" ->[softmax, slice]-> "smoking" 
        """
        # [b,c,h,w]x3 -> [b3,c,h,w]
        x0 = self.proxy_first_conv(input0)
        x1 = self.proxy_first_conv(input1)
        x2 = self.proxy_first_conv(input2)

        face_feat = self.backbone_face(x0)*self._feat_weight(x0)
        hand1_feat = self.backbone_hand(x1)*self._feat_weight(x1)
        hand2_feat = self.backbone_hand(x2)*self._feat_weight(x2)

        feat = torch.cat([face_feat, hand1_feat, hand2_feat], dim=1)
        feat_pool = self.avgpool(feat)#[b,c,1,1]
        output = self.fc(feat_pool)#[b,c,1,1]
        output = output.reshape(-1, self.num_classes) #[b,c]
        output = F.softmax(output, dim=-1)
        # print(output.shape)
        output = output[:,1:2]
        # print(output.shape)
        return output


class SmokingModel_1_FACE_MLU(nn.Module):
    """
    preprocess:
    bgr2rgb
    (x-mean)/std
    py_mean = [123.675, 116.28, 103.53]
    py_std = [58.395, 57.12, 57.375]
    """
    def __init__(self, num_classes=2, model_name='resnet34'):
        super(SmokingModel_1_FACE_MLU, self).__init__()
        self.proxy_first_conv = ProxyConvModule()
        if model_name not in model_dict:
            raise NotImplementedError('model:{} is not defined'.format(model_name))
        self.backbone_face = model_dict[model_name]() # n c w h
        self.fc = nn.Conv2d(512*3, num_classes, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_classes = num_classes


    def forward(self, input):
        """
        :param input: [b,c,h,w]
        :return: [b,2] "non-smoking, smoking" ->[softmax, slice]-> "smoking" 
        """
        # [b,c,h,w] -> [b,c,h,w]
        face_input = self.proxy_first_conv(input)
        face_feat = self.backbone_face(face_input)
        hand_feat = torch.zeros_like(face_feat)
        feat = torch.cat([face_feat, hand_feat, hand_feat], dim=1)
        feat_pool = self.avgpool(feat)#[b,c,1,1]
        output = self.fc(feat_pool)#[b,c,1,1]
        output = output.reshape(-1, self.num_classes) #[b,c]
        output = F.softmax(output, dim=-1)
        # print(output.shape)
        output = output[:,1:2]
        # print(output.shape)
        return output
    