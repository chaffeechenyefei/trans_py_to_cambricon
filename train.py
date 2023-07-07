import os
import time
import numpy as np
import argparse
import torch
import torch.nn as nn

from models.unet import UNet3D, UNet2DShift, UNetResNet18_Shift, UNetResNet18, UNetResNet18_BN
from dataloader.dataloader import zDataLoader
from dataloader.dataset import VirtualMOD_Dataset
from checkpoint.checkpoint import CheckpointMgr
from models.loss import FocalLoss_BCE
pj = os.path.join

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bgpath',default='/project/data/coco2017/train2017')
    parser.add_argument('--output',default='./pth/UNet2DShift/')
    parser.add_argument('--lr',default=0.001,type=float)
    parser.add_argument('--max_epoch',default=100,type=int)
    parser.add_argument('--batchsize',default=6,type=int)
    parser.add_argument('--view_interval',default=50,type=int)
    parser.add_argument('--ckpt_interval',default=1000,type=int)
    parser.add_argument('--t',default=8, type=int)
    args = parser.parse_args()
    return args

"""
nohup python -u train.py --t 3 --batchsize 24 >x2dshift.out 2>&1 &
nohup python -u train.py --t 2 --batchsize 24 --output ./pth/UNet2DShift/bs2/ >x2dshift.out 2>&1 &

nohup python -u train.py --t 2 --batchsize 24 --output ./pth/UNetResNet18_Shift/bs2/ >x2dshift.out 2>&1 &
nohup python -u train.py --t 2 --batchsize 48 --output ./pth/UNetResNet18/bs2/ >x2dshift.out 2>&1 &
nohup python -u train.py --t 2 --batchsize 48 --output ./pth/UNetResNet18_BN/bs2/ >x2dshift.out 2>&1 &
"""
def train(args):
    n_cuda_device = torch.cuda.device_count()
    # n_cuda_device = 1
    T = args.t
    dataset_train = VirtualMOD_Dataset(bg_path=args.bgpath,fg_path=args.bgpath,ext='.jpg', T=T)
    dataloader = zDataLoader(imgs_per_gpu=args.batchsize,workers_per_gpu=8 if n_cuda_device > 1 else 16,num_gpus=n_cuda_device,dist=False,shuffle=True,pin_memory=True,verbose=True)(dataset_train)

    # model = UNet3D(n_channels=3,n_classes=1,bilinear=True)
    # model = UNet2DShift(n_channels=3, n_classes=1, n_segment=T)
    # model = UNetResNet18_Shift(n_classes=1, n_segment=T)
    model = UNetResNet18_BN(n_classes=1, n_segment=T, use_bn=False)
    model = model.initial(pretrained_weights='pth/UNetResNet18/bs2/milestone/model_175.pth')

    trainable_params = model.get_parameters()
    optimizer = torch.optim.SGD(trainable_params,lr=args.lr,momentum=0.9)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=2.0*torch.ones(1))
    criterion = FocalLoss_BCE(gamma=2.0, alpha=0.4)
    checkpoint_op = CheckpointMgr(ckpt_dir=args.output)
    checkpoint_op.load_checkpoint(model,map_location='cpu')
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 25, 30, 35], gamma=0.5,
                                                     last_epoch=-1)
    model = model.cuda()
    criterion = criterion.cuda()
    if n_cuda_device > 1:
        model = nn.DataParallel(model)
    model.train()


    for epoch in range(args.max_epoch):
        model.train()
        lr = optimizer.param_groups[0]['lr']
        epoch_loss, epoch_tm, iter_tm = [],0,[]
        tm = time.time()
        for ind,batch in enumerate(dataloader):
            # print('{:.1f}%'.format(ind/len(dataloader)*100), end='\r')
            process_line = ind/len(dataloader)*100
            imgs = batch['X']#[b,c,t,h,w]
            assert len(imgs.shape) == 5, 'batch[X]={}'.format(imgs.shape)
            masks = batch['y']#[b,h,w]

            imgs = imgs.cuda()
            masks = masks.cuda()
            pred = model(imgs)#[b,h,w]

            loss = criterion(pred.reshape(-1),masks.reshape(-1))
            epoch_loss.append(loss.data.cpu().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tm = time.time() - tm
            epoch_tm += tm
            iter_tm.append(tm)
            if ind%args.view_interval==0:
                print('Epoch:{:3d}[{:.1f}%], iter:{}, loss:{:.3f}[{:.3f}], lr:{:.5f},{:.2f}s/iter'.format(
                    epoch, process_line ,ind, loss.item(), np.array(epoch_loss).mean(), lr ,np.array(iter_tm).mean()
                ))
            iter_tm = []
            tm = time.time()
            # if ind%args.ckpt_interval==0 and ind >= args.ckpt_interval:
            #     print('Saving...')
            #     checkpoint_op.save_checkpoint(model=model.module if n_cuda_device > 1 else model, verbose=False)
        # outer

        print('Epoch:{:3d},total_loss: {:.4f}, lr:{:.5f},{:.2f}s'.format(
            epoch, np.array(epoch_loss).mean(), lr, epoch_tm
        ))
        scheduler.step()
        print('Saving...')
        checkpoint_op.save_checkpoint(model=model.module if n_cuda_device > 1 else model, verbose=False)





if __name__ == '__main__':
    args = arg_parse()
    train(args)


