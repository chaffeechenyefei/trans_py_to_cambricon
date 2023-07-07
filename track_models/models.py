import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image


class Resnet18_MLU(nn.Module):
    def __init__(self):
        super(Resnet18_MLU,self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)

        fc = nn.Sequential(*[nn.Linear(512, 256),
                                   nn.BatchNorm1d(256),
                                   nn.Dropout()
                                   ])
        conv1.apply(weights_init_kaiming)
        self.model.conv1 = conv1
        fc.apply(weights_init_kaiming)
        self.model.fc = fc

    def forward(self,x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        # xnorm = torch.norm(x, p=2, dim=1, keepdim=True)
        xnorm = torch.sqrt( torch.sum( x**2, dim=1, keepdim=True ) ) + 1e-4
        x = x/xnorm
        return x

class Resnet18(nn.Module):
    def __init__(self,num_cls,test=False):
        super(Resnet18,self).__init__()
        self.test=test
        self.model = torchvision.models.resnet18(pretrained=True)
        conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)

        fc = nn.Sequential(*[nn.Linear(512, 256),
                                   nn.BatchNorm1d(256),
                                   nn.Dropout()
                                   ])
        self.classifier = nn.Sequential(*[nn.Linear(256, num_cls)])
        conv1.apply(weights_init_kaiming)
        self.model.conv1 = conv1
        fc.apply(weights_init_kaiming)
        self.model.fc = fc
        # self.classifier.apply(weight_init)
        # weights_init_kaiming(self.fc)
        # weight_init(self.classifier)
        # basic_net = nn.Sequential(*list(basic_net.children()+fc_layer))
    def forward(self,x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        if self.test:
            xnorm = torch.norm(x, p=2, dim=1, keepdim=True)
            x = x/xnorm
        else:
            x = self.classifier(x)
        return x


def weight_init(module):
    for item in module.children():
        if isinstance(item, nn.Linear):
            nn.init.normal_(item.weight.data, std=0.001)
            nn.init.constant_(item.bias.data, 0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0.0)




def data_preprocess(im_path):
    image = Image.open(im_path).convert("RGB")
    val_transform = T.Compose([
        T.Resize((128, 64), interpolation=3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#123.675,116.28,103.53; 58.395,57.12,57.375
    ])
    image = val_transform(image)
    return image


def extract_feature(im_path,ckpt):
    net = Resnet18(1310,test=True)
    net.load_state_dict(torch.load(ckpt),strict=False)
    net.eval()

    image = data_preprocess(im_path)
    feature = net.forward(image)
    return feature