"""
mlu head
"""
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
"""
yolov5
"""
# from diffunet_models.unet_utils import DIFFUNet4_MLU
from smoking_models.models import SmokingModel_1_MIMO_MLU
from checkpoint.checkpoint import CheckpointMgr

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_date',default='199')
    parser.add_argument('--w',default=736, type=int)
    parser.add_argument('--h',default=416, type=int)
    parser.add_argument('--t',default=3, type=int)
    parser.add_argument('--mpath',default='pth/UNetResNet18_BN/bs2/milestone')
    # Check
    parser.add_argument('--check', action='store_true')
    # Quant and Infer
    parser.add_argument('--data',help='data path to the images used for quant')
    parser.add_argument('--ext',default='.jpg')

    parser.add_argument('--mlu', default=True, type=str2bool,
                        help='Use mlu to train model')
    parser.add_argument('--jit', default=True, type=str2bool,
                        help='Use jit for inference net')
    parser.add_argument('--quantization', default=False, type=str2bool,
                        help='Whether to quantize, set to True for quantization')

    parser.add_argument("--batch_size", dest="batch_size", help="batch size for one inference.",
                        default=1, type=int)

    # Advance
    parser.add_argument("--quantized_mode", dest='quantized_mode', help=
    "the data type, 0-float16 1-int8 2-int16, default 1.",
                        default=1, type=int)
    parser.add_argument("--half_input", dest='half_input', help=
    "the input data type, 0-float32, 1-float16/Half, default 1.",
                        default=1, type=int)

    # Useless
    parser.add_argument('--core_number', default=4, type=int,
                        help='Core number of mfus and offline model with simple compilation.')
    parser.add_argument('--mcore', default='MLU270', type=str,
                        help="Set MLU Architecture")

    args = parser.parse_args()

    if args.check:
        print('==Checking==')
        cpu_res = np.load('cpu_pred.npy')
        mlu_res = np.load('mlu_pred.npy')
        quant_res = np.load('cpu_quant_pred.npy')[:mlu_res.shape[0],:,:]

        print('cpu:', cpu_res.shape)
        print('mlu', mlu_res.shape)
        print('quant:', quant_res.shape)

        diff = np.sqrt(np.sum((cpu_res - mlu_res)**2))/cpu_res.shape[0]
        print('Sqrt Diff cpu vs mlu: {:.3f}'.format(diff))

        diff = np.sqrt(np.sum((quant_res - mlu_res)**2))/quant_res.shape[0]
        print('Sqrt Diff quant vs mlu: {:.3f}'.format(diff))

        diff = np.sqrt(np.sum((quant_res - cpu_res)**2))/cpu_res.shape[0]
        print('Sqrt Diff quant vs cpu: {:.3f}'.format(diff))

        print('==Done==')
        exit(0)

    """
    Environment
    """
    ct.set_core_number(args.core_number)
    ct.set_core_version(args.mcore)
    print("batch_size is {:d}, core number is {:d}".format(args.batch_size, args.core_number))
    dtype = 'int8'
    if args.quantized_mode == 0:
        dtype = 'float16'
    elif args.quantized_mode == 1:
        dtype = 'int8'
    elif args.quantized_mode == 2:
        dtype = 'int16'
    else:
        pass
    print('using dtype = {}'.format(dtype))
    """
    ATT. Parameters
    """
    model_framework = 'smoking-r34'
    model_path = args.mpath
    print('using model path: {}'.format(model_path))
    model_date = args.model_date
    model_name = 'model_{}_no_serial.pth'.format(model_date)#old format unzipped
    print('@@Model pth = {}'.format(model_name))
    model_online_fullname = pj(model_path, 'mlu_mimo_{}_{}_{}_{:d}x{:d}.pth'.format(dtype,model_framework,model_date,
                                                                               args.w,args.h))
    IMG_SIZE = [args.w,args.h]  # [w,h]
    T = args.t
    """
    在模型内做归一化，使用first conv目的是增加吞吐，实现mlu端到端的对接
    """
    # py_mean = [0, 0, 0]
    # py_std = [1, 1, 1]
    py_mean = [123.675, 116.28, 103.53]
    py_std = [58.395, 57.12, 57.375]
    mlu_mean_std = trans_mean_std_py2mlu(py_mean,py_std)
    mlu_mean = mlu_mean_std['mean']
    mlu_std = mlu_mean_std['std']
    print("")

    """
    Import data
    input_img = [(n,3,H,W)]
    """
    image_list = [ pj(args.data,c) for c in os.listdir(args.data) if c.endswith(args.ext) ]
    K = min([len(image_list),args.batch_size])
    image_list = image_list[:K]
    print('sampled %d data'%len(image_list))
    print(image_list[0])
    input_img = [cv2.imread(c) for c in image_list]

    """
    当且仅当使用mlu做推理，且使用first_conv策略时适配
    """
    # use_first_conv = False
    # flag_preprocess_mlu = True #不做归一化， 在模型内部做
    use_first_conv = True
    flag_preprocess_mlu = False
    if args.mlu and use_first_conv:
        flag_preprocess_mlu = True
    """
    data = [(n,c,h,w)]
    [n,c,h,w]xB
    """
    input_img = input_img*5
    data = [preprocess_tsn_bt_c_h_w(c , dst_size=IMG_SIZE , mean=py_mean, std = py_std, mlu=flag_preprocess_mlu, T=T) for c in input_img]
    data = [c[0] for c in data]
    print('len of data: %d'%len(data))
    """
    [n,c,h,w]xN -> [bt,c,h,w]
    """
    data = torch.cat(data,dim=0)
    """
    shuffle
    """
    data = data[torch.randperm(data.size(0))]
    data = data.reshape(-1,T,3,IMG_SIZE[1],IMG_SIZE[0])#[b,t,c,h,w]

    print('data shape =',data.shape)
    if args.mlu:
        if args.half_input == 1:
            print('using half input')
            data = data.type(torch.HalfTensor)
        elif args.half_input == 2:
            data = data.type(torch.IntTensor)
        data = data.to(ct.mlu_device())

    """
    Import pytorch model on cpu first
    """
    print('==pytorch==')
    use_device = 'cpu'
    loading = True if not args.mlu else False
    model = SmokingModel_1_MIMO_MLU()

    if loading:
        print('==loading==')
        save_dir = model_path
        checkpoint_op = CheckpointMgr(ckpt_dir=save_dir)
        checkpoint_op.load_checkpoint(model=model, ckpt_fpath=pj(save_dir, model_name), warm_load=False)

    print('==end==')
    # print(model.modality)
    model = model.eval().float()

    if args.quantization:
        print('doing quantization on cpu')
        use_avg = False if data.shape[0] == 1 else True
        print('fisrtconv: ', use_first_conv)
        qconfig = { 'per_channel':True, 'firstconv':use_first_conv ,'mean': mlu_mean , 'std': mlu_std}
        model_quantized = mlu_quantize.quantize_dynamic_mlu(model, qconfig, dtype=dtype, gen_quant = True)
        print('data.shape=',data.shape)
        preds = model_quantized(data[:,0,...],data[:,1,...],data[:,2,...])
        torch.save(model_quantized.state_dict(), model_online_fullname )
        print("int8 quantization end!")

        _preds0 = fetch_cpu_data(preds[0], args.half_input)
        # _preds1 = fetch_cpu_data(preds[1], args.half_input)

        print('saving', _preds0.shape)
        # print(preds.shape) #[b,d]
        # np.save('cpu_quant_pred.npy', _preds1)
    else:
        if not args.mlu:
            print('doing cpu inference')
            with torch.no_grad():
                preds = model(data)

                _preds0 = fetch_cpu_data(preds[0], args.half_input)
                # _preds1 = fetch_cpu_data(preds[1], args.half_input)

                # print('saving', _preds1.shape)
                print(_preds0.shape)
                # np.save('cpu_pred.npy', _preds1)
            print("cpu inference finished!")
        else:
            print('doing mlu inference')
            model = mlu_quantize.quantize_dynamic_mlu(model)
            checkpoint = torch.load(model_online_fullname, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
            # model.eval().float()
            model = model.eval().float().to(ct.mlu_device())
            if args.jit:
                print('using jit inference')
                randinput = torch.rand(T, 3, IMG_SIZE[1], IMG_SIZE[0])*255
                if args.half_input == 1:
                        randinput = randinput.type(torch.HalfTensor)
                randinput = randinput.to(ct.mlu_device())

                traced_model = torch.jit.trace(model, randinput, check_trace=False)
                # print(traced_model.graph)
                print('start inference')
                preds = traced_model(data)
                print('end inference')
                _preds0 = fetch_cpu_data(preds[0], args.half_input)
                # _preds1 = fetch_cpu_data(preds[1], args.half_input)
                # print('saving', _preds1.shape)
                # np.save('mlu_jit_pred.npy', _preds1)
                print("mlu inference finished!")
            else:
                print('using layer by layer inference')
                data = data.to(ct.mlu_device())
                preds = model(data)
                print('done')

                _preds0 = fetch_cpu_data(preds[0], args.half_input)
                # _preds1 = fetch_cpu_data(preds[1], args.half_input)

                # print('saving', _preds1.shape)
                print(_preds0.shape)
                # np.save('mlu_pred.npy', _preds1)
                print("mlu inference finished!")