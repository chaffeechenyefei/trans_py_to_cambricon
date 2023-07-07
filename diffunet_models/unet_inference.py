import os
import cv2
import json
import tqdm
import numpy as np
from diffunet_models.unet_utils import Inference, DIFFUNet4, img_preprocess

val_dir = '/mnt/projects/MovingDetection/dataset/testset/validation/'
mask_dir = '/mnt/projects/MovingDetection/dataset/testset/validation_mask/'
save_dir = '/mnt/projects/MovingDetection/dataset/testset/results/'

model_fn = 'diffUnet4_736'
use_cuda = True
model_fp = '/mnt/projects/MovingDetection/weights/diffUnet4_224/bestmodel_104_0.289656.pth'
img_size = (736, 416)
model = DIFFUNet4()
inference = Inference(model=model, ckpt_path=model_fp, use_cuda=use_cuda)

save_video = True
iou_thresh = 0.2
prec_thresh = 0.2
dir_list = ['tv04', 'tv05', 'tv07', 'cvmart01']
iou_results, prec_results = [], []

for video_fn in dir_list:
    video_dir = os.path.join(val_dir, video_fn)
    frame_list = os.listdir(video_dir)
    img = cv2.imread(os.path.join(video_dir, '0.jpg'))
    img_h, img_w = img.shape[:2]

    if save_video:
        vid_fps = 28
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        filename = video_fn + f'_{model_fn}.mp4'
        vid = cv2.VideoWriter(os.path.join(save_dir, filename), fourcc, vid_fps, (2 * img_w, img_h))

    precision, iou_list = 0, []

    bar = range(len(frame_list) - 1)
    bar = tqdm.tqdm(bar)
    for frame_id in bar:
        img1_fp = os.path.join(video_dir, f'{frame_id}.jpg')
        img1 = cv2.imread(img1_fp)

        img2_fp = os.path.join(video_dir, f'{frame_id + 1}.jpg')
        img2 = cv2.imread(img2_fp)

        inputs = img_preprocess([img1, img2], dst_size=img_size, use_cuda=use_cuda)

        pred = inference.inference(inputs)
        pred = (pred > iou_thresh).astype(np.uint8)
        pred = cv2.resize(pred, (img_w, img_h))

        mask = np.zeros((img_h, img_w), np.uint8)
        mask_fp = os.path.join(mask_dir, video_fn, f'{frame_id}.json')
        if os.path.isfile(mask_fp):
            with open(mask_fp, 'r') as f:
                mask_data = json.load(f)
                mask_points = np.array(mask_data['shapes'][0]['points'])
                cv2.fillPoly(mask, mask_points.astype(np.int64).reshape(1, -1, 2), 1)

        if np.sum(mask):
            intersection = pred * mask
            iou = np.sum(intersection) / (np.sum(pred) + np.sum(mask) - np.sum(intersection))
            iou_list.append(iou)

            if np.mean(pred) >= prec_thresh:
                precision += 1

        else:
            if np.mean(pred) < prec_thresh:
                precision += 1

        bar.set_description('video: {}, iou: {:.4f}, precision: {:.4f}'.format(video_fn,
                                                                               np.mean(iou_list) if len(
                                                                                   iou_list) else 0,
                                                                               precision / (frame_id + 1)))

        if save_video:
            img_masked = img1.copy()
            img_masked = img_masked.astype(np.float32)
            roi = mask.astype(np.bool)
            img_masked[~roi] *= 0.5
            img_masked[roi] = 255
            img_masked = img_masked.astype(np.uint8)

            img_preded = img1.copy()
            img_preded = img_preded.astype(np.float32)
            roi = pred.astype(np.bool)
            img_preded[~roi] *= 0.5
            img_preded[roi] = 255
            img_preded = img_preded.astype(np.uint8)

            img = np.concatenate([img_masked, img_preded], axis=1)
            vid.write(img)

    iou_results.append(np.mean(iou_list))
    prec_results.append(precision / (len(frame_list) - 1))
    vid.release() if save_video else None

print('iou: {:.4f}, precision: {:.4f}'.format(np.mean(iou_results), np.mean(prec_results)))