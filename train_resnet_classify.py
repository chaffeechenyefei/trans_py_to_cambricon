import os
import time
import numpy as np
import argparse
import torch
import torch.nn as nn

from models.resnet import fit_resnet34
from dataloader.dataloader import zDataLoader
from dataloader.dataset import DunningsFire_Dataset
from checkpoint.checkpoint import CheckpointMgr
pj = os.path.join

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--bgpath',default='/project/data/coco2017/train2017')
    parser.add_argument('--trainpath', default='/project/data/Fire/dunnings-2018/images-224x224/train/')
    parser.add_argument('--valpath',default='/project/data/Fire/dunnings-2018/images-224x224/test/')
    parser.add_argument('--output',default='./pth/ResNet34_Fire/')
    parser.add_argument('--lr',default=0.001,type=float)
    parser.add_argument('--max_epoch',default=100,type=int)
    parser.add_argument('--batchsize',default=6,type=int)
    parser.add_argument('--view_interval',default=50,type=int)
    parser.add_argument('--ckpt_interval',default=1000,type=int)
    args = parser.parse_args()
    return args

"""
nohup python -u train_resnet_classify.py --batchsize 128 >xresnet.out 2>&1 &
"""
def val(args):
    from utils.evaluation import print_precision_wiz_recall
    # n_cuda_device = torch.cuda.device_count()
    n_cuda_device = 1
    model = fit_resnet34(pretrained_path=None, class_num=2)
    checkpoint_op = CheckpointMgr(ckpt_dir=args.output)
    checkpoint_op.load_checkpoint(model, map_location='cpu')
    model = model.cuda()

    dataset_val = DunningsFire_Dataset(datapath=args.valpath, bg_path=None,mode='test', use_additional_data=True)
    dataloader = zDataLoader(imgs_per_gpu=args.batchsize, workers_per_gpu=8 if n_cuda_device > 1 else 16,
                             num_gpus=n_cuda_device, dist=False, shuffle=False, pin_memory=True, verbose=True)(
        dataset_val)
    model.eval()
    y_scores = []
    y_truths = []
    for ind, batch in enumerate(dataloader):
        imgs = batch['X']  # [b,c,h,w]
        labels = batch['y']  # [b]

        imgs = imgs.cuda()
        pred = model.inference(imgs)  # [b,nc], nc=2

        y_score = pred.cpu().data.numpy()[:,1] #[b]
        y_truth = np.array(labels)

        y_scores.append(y_score)
        y_truths.append(y_truth)

    y_scores = np.concatenate(y_scores,axis=0)
    y_truths = np.concatenate(y_truths,axis=0)
    print_precision_wiz_recall(y_true=y_truths, y_score=y_scores, recall=[0.99,0.95,0.9])
    model.train()



def train(args):
    n_cuda_device = torch.cuda.device_count()
    # n_cuda_device = 1
    dataset_train = DunningsFire_Dataset(datapath=args.trainpath, bg_path=args.bgpath,bg_ratio=0.5 ,mode='train')
    dataloader = zDataLoader(imgs_per_gpu=args.batchsize,workers_per_gpu=8 if n_cuda_device > 1 else 16,num_gpus=n_cuda_device,dist=False,shuffle=True,pin_memory=True,verbose=True)(dataset_train)

    """
    model
    """
    model = fit_resnet34(pretrained_path='/project/pytorch_official_weights/resnet34-333f7ec4.pth',class_num=2)
    """
    optimizer
    """
    trainable_params = model.get_parameters()
    optimizer = torch.optim.SGD(trainable_params,lr=args.lr,momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    checkpoint_op = CheckpointMgr(ckpt_dir=args.output)
    checkpoint_op.load_checkpoint(model,map_location='cpu')
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30 , 40], gamma=0.5,
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
            imgs = batch['X']#[b,c,h,w]
            labels = batch['y']#[b]
            labels = torch.LongTensor(labels)

            # print(imgs.shape, labels.shape)

            imgs = imgs.cuda()
            labels = labels.cuda()
            preds = model(imgs)#[b,nc]

            loss = criterion(preds,labels)
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

        print('Epoch:{:3d},total_loss: {:.3f}, lr:{:.5f},{:.2f}s'.format(
            epoch, np.array(epoch_loss).mean(), lr, epoch_tm
        ))
        scheduler.step()
        print('Saving...')
        checkpoint_op.save_checkpoint(model=model.module if n_cuda_device > 1 else model, verbose=False)
        """VAL"""
        print('#'*20)
        print('EVAL')
        print('#' * 20)
        val(args)
        print('#'*20)
        print('#' * 20)




if __name__ == '__main__':
    args = arg_parse()
    val(args)
    if not args.test:
        train(args)


