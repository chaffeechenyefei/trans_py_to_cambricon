import os
import time
import numpy as np
import argparse
import torch
import torch.nn as nn
import cv2

from models.unet import UNet2D
from dataloader.dataloader import zDataLoader
from dataloader.dataset import WaterBasicDataset
from checkpoint.checkpoint import CheckpointMgr
pj = os.path.join

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath',default='/project/data/kaggle_water_segmentation/water_tmp/water_tmp')
    parser.add_argument('--result', default='/project/data/kaggle_water_segmentation/water_tmp/result')
    parser.add_argument('--output',default='./pth/UNet2D_Water/')
    parser.add_argument('--lr',default=0.001,type=float)
    parser.add_argument('--max_epoch',default=100,type=int)
    parser.add_argument('--batchsize',default=6,type=int)
    parser.add_argument('--view_interval',default=50,type=int)
    parser.add_argument('--ckpt_interval',default=1000,type=int)
    args = parser.parse_args()
    return args


def val(args):
    print('#'*10, 'EVAL' , '#'*10)
    if not os.path.exists(args.result):
        os.makedirs(args.result, exist_ok=True)
    # n_cuda_device = torch.cuda.device_count()
    n_cuda_device = 1
    dataset_test = WaterBasicDataset(datapath=args.datapath)
    dataloader = zDataLoader(imgs_per_gpu=args.batchsize,workers_per_gpu=8 if n_cuda_device > 1 else 16,
                             num_gpus=n_cuda_device,dist=False,shuffle=False,pin_memory=True,verbose=True)(dataset_test)

    model = UNet2D(n_channels=3, n_classes=1)

    checkpoint_op = CheckpointMgr(ckpt_dir=args.output)
    checkpoint_op.load_checkpoint(model,map_location='cpu')

    model = model.cuda()
    if n_cuda_device > 1:
        model = nn.DataParallel(model)
    model.eval()

    cnt = 0
    for ind,batch in enumerate(dataloader):
        process_line = ind/len(dataloader)*100
        imgs = batch['X']#[b,c,h,w]
        ori_imgs = batch['img']#[b,h,w,c]
        imgnames = batch['name']

        imgs = imgs.cuda()
        preds = model.inference(imgs)#[b,h,w]

        for imgname, img,pred in zip(imgnames, ori_imgs, preds):
            img = img.cpu().data.numpy().astype(np.uint8) #[h,w,3] uint8
            pred = pred.cpu().data.numpy() #[h,w] fp32

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[...,-1] = (hsv[...,-1]*pred).astype(np.uint8)
            img2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            img = np.concatenate([img2,img], axis=1)
            cv2.imwrite( pj(args.result,'{}.png'.format(imgname)), img)

            cnt += 1
    print('#' * 10, 'EVAL END', '#' * 10)


if __name__ == '__main__':
    args = arg_parse()
    val(args)


