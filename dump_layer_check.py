import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize
import argparse
import torch
import cv2
import numpy as np
import random,math
import os
torch.set_grad_enabled(False)
pj = os.path.join

from torch_mlu.core.utils import dump_utils

"""
yolov5
"""

def str2bool(v):
     return v.lower() in ("yes", "true", "t", "1")


def trans_mean_std_py2mlu(mean, std):
    """
    py: x = (x-mean)/std
    mlu: x = (x/255-m)/s = (x-255m)/(255s)
    :return: 
    """
    m = [ c/255 for c in mean]
    s = [ c/255 for c in std]
    return {
        "mean": m,
        "std": s
    }

def trans_mean_std_mlu2py(mean,std):
    """
    py: x = (x-m)/s
    mlu: x = (x/255-mean)/std = (x-255mean)/(255std)
    :return: 
    """
    s = [ 255*c for c in std ]
    m = [ 255*c for c in mean ]
    return {
        "mean":m,
        "std":s
    }

def fetch_cpu_data(x,use_half_input=False):
    if use_half_input:
        output = x.cpu().type(torch.FloatTensor)
    else:
        output = x.cpu()
    return output.detach().numpy()

def preprocess_tsn_bt_c_h_w(img_cv, dst_size ,mean, std, T = 8, mlu=False):
    """
    :param img_cv: 
    :param dst_size: 
    :param mean: 
    :param std: 
    :param T: 
    :param mlu: 
    :return: (t,c,h,w) 
    将错就错, BGR用了RGB的归一化参数
    """
    h,w = img_cv.shape[:2]
    dstW,dstH = dst_size
    aspect_ratio = min([dstW/w,dstH/h])
    _h = min([ int(h*aspect_ratio), dstH])
    _w = min([ int(w*aspect_ratio), dstW])

    # print(dstH,dstW,h,w, _h,_w)

    padh = dstH - _h
    padw = dstW - _w

    left_w_pad = int(padw/2)
    up_h_pad = int(padh/2)

    img_resized = cv2.resize(img_cv, (_w,_h) )

    img_dst = np.zeros([dstH,dstW,3],np.uint8)
    img_dst[up_h_pad:up_h_pad+_h,left_w_pad:left_w_pad+_w] = img_resized*1

    img_dst = img_dst.transpose(2, 0, 1) #[c,h,w]
    img_dst = torch.from_numpy(img_dst)
    img_dst = img_dst.float()  # uint8 to fp16/32

    if not mlu:
        mean = torch.FloatTensor(mean).reshape(-1,1,1)
        std = torch.FloatTensor(std).reshape(-1,1,1)
        img_dst -= mean
        img_dst /= std

    img_dst = torch.stack( [img_dst]*T, 0) #[c,h,w]xT -> [T,c,h,w]
    return img_dst, aspect_ratio



parser = argparse.ArgumentParser()
# Quant and Infer
parser.add_argument('--data',help='data path to the images used for quant')
parser.add_argument('--ext',default='.jpg')
parser.add_argument("--batch_size", dest="batch_size", help="batch size for one inference.",
                    default=1, type=int)

# Advance
parser.add_argument("--half_input", dest='half_input', help=
"the input data type, 0-float32, 1-float16/Half, default 1.",
                    default=1, type=int)

# Useless
parser.add_argument('--core_number', default=4, type=int,
                    help='Core number of mfus and offline model with simple compilation.')
parser.add_argument('--mcore', default='MLU270', type=str,
                    help="Set MLU Architecture")

args = parser.parse_args()
"""
Environment
"""
ct.set_core_number(args.core_number)
ct.set_core_version(args.mcore)
print("batch_size is {:d}, core number is {:d}".format(args.batch_size, args.core_number))
"""
ATT. Parameters
"""
# model_path = './pth/UNetResNet18/bs2/milestone'
# mlu_model_path = pj( model_path, 'mlu_int8_unetResNet18_175_224x224.pth')
# torch_model_path = pj(model_path, 'model_175_no_serial.pth')
model_path = './pth/UNetResNet18_BN/bs2/milestone'
mlu_model_path = pj( model_path, 'mlu_int8_unetResNet18_bn_110_224x224.pth')
torch_model_path = pj(model_path, 'model_110_no_serial.pth')


IMG_SIZE = [224, 224]  # [w,h]
py_mean = [123.675, 116.28, 103.53]
py_std = [58.395, 57.12, 57.375]
mlu_mean_std = trans_mean_std_py2mlu(py_mean, py_std)
mlu_mean = mlu_mean_std['mean']
mlu_std = mlu_mean_std['std']
print("")

"""
Import data
"""
image_list = [pj(args.data, c) for c in os.listdir(args.data) if c.endswith(args.ext)]
image_list.sort()
K = min([len(image_list), args.batch_size])
image_list = image_list[:K]
print('sampled %d data' % len(image_list))
print(image_list[0])
input_img = [cv2.imread(c) for c in image_list]
# set mlu=False to always trigger normalization(mean,std)

"""
data = [(c,th,w)]
[t,c,h,w]xN
"""
torch_data = [preprocess_tsn_bt_c_h_w(c, dst_size=IMG_SIZE, mean=py_mean, std=py_std, mlu=False, T=2) for c in
        input_img]
torch_data = [c[0] for c in torch_data]
print('len of data: %d' % len(torch_data))

mlu_data = [preprocess_tsn_bt_c_h_w(c, dst_size=IMG_SIZE, mean=py_mean, std=py_std, mlu=True, T=2) for c in
        input_img]
mlu_data = [c[0] for c in mlu_data]
"""
[t,c,h,w]xN -> [bt,c,h,w]
"""
torch_data = torch.cat(torch_data, dim=0)
mlu_data = torch.cat(mlu_data, dim=0)
print('mlu_data =', mlu_data.shape)
print('torch_data =', torch_data.shape)

# if args.mlu:
#     if args.half_input:
#         data = data.type(torch.HalfTensor)
#     data = data.to(ct.mlu_device())

from models.unet import UNetResNet18_BN, UNetResNet18_BN_MLU
"""
Import pytorch model on cpu first
"""
print('==pytorch==')
use_device = 'cpu'
model = UNetResNet18_BN_MLU(n_classes=1,n_segment=2)
ckpt = torch.load(torch_model_path, map_location='cpu')
model.load_state_dict(ckpt,strict=False)
print('==loaded==')
model = model.eval().float()


dump_utils.register_dump_hook(model, start='layer1.0.conv1', end='sigmoid')#  start='layer1.0.conv1', end='sigmoid'
pred = model(torch_data)
pred_cpu = fetch_cpu_data(pred)
dump_utils.save_data('dump/',"cpu")
print('==end==')


"""
Import mlu torch model on mlu then
"""
print('==mlu==')
model_mlu = UNetResNet18_BN_MLU(n_classes=1,n_segment=2)
model_mlu = mlu_quantize.quantize_dynamic_mlu(model_mlu)
ckpt = torch.load(mlu_model_path, map_location='cpu')
model_mlu.load_state_dict(ckpt, strict=False)
model_mlu = model_mlu.eval().float().to(ct.mlu_device())
mlu_data = mlu_data.to(ct.mlu_device())
print('==loaded==')
dump_utils.register_dump_hook(model_mlu, start='layer1.0.conv1', end='sigmoid') #  start='layer1.0.conv1', end='sigmoid' start='self.layer1', end='self.sigmoid')
print('==registed==')
pred = model_mlu(mlu_data)
print('==predicted==')
pred_mlu = fetch_cpu_data(pred)
dump_utils.save_data('dump/',"mlu")
print('==end==')

print('==diffing==')
dump_utils.diff_data("dump/dump_cpu_data.pth", "dump/dump_mlu_data.pth")

diff = np.sqrt(np.sum((pred_cpu[0,:,:] - pred_mlu[0,:,:])**2))
print(diff)
print('==end==')