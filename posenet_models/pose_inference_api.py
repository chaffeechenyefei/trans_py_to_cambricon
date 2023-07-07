import torch
import cv2
import numpy as np


'''
"keypoints": {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
},
"skeleton": [
    [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
    [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
'''

class PoseInference:

    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        from deploy import pose_resnet_slim
        model = pose_resnet_slim.pose_resnet18()
        ckpt_path = './model_best.pth'
        save_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(save_dict)
        model.eval()
        return model

    def _img_preprocess(self, img_cv2, wh=(192, 256)):
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        wh_rescale_ratio = (img_cv2.shape[1] / wh[0], img_cv2.shape[0] / wh[1])
        img_cv2 = cv2.resize(img_cv2, (192, 256))
        img_data = np.array(img_cv2, np.float32)
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        img_data = (img_data / 255.0 - mean) / std
        x = np.expand_dims(img_data, axis=0)
        x = np.transpose(x, axes=(0, 3, 1, 2))
        x = torch.from_numpy(x)
        return x, wh_rescale_ratio

    @staticmethod
    def get_max_preds(batch_heatmaps):
        '''
        get predictions from score maps
        heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
        '''
        assert isinstance(batch_heatmaps, np.ndarray), \
            'batch_heatmaps should be numpy.ndarray'
        assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        batch_size = batch_heatmaps.shape[0]
        num_joints = batch_heatmaps.shape[1]
        width = batch_heatmaps.shape[3]
        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask
        return preds, maxvals

    def predict(self, img_cv2):
        x, wh_rescale_ratio = self._img_preprocess(img_cv2)
        y = self.model(x)
        preds, maxvals = self.get_max_preds(y.cpu().detach().numpy())
        # preds: shape=[1, 17, 2], batchsize=1, num_points=17, len([x,y])=2
        return preds[0] * wh_rescale_ratio

    @staticmethod
    def draw_img_with_joints(img_cv2, joints, joints_vis):
        '''
        joints: [n_joints, 3]
        '''
        sks = np.array([
            [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 6],
            [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12],
            [11, 13], [13, 15], [12, 14], [14, 16]])
        for sk in sks:
            # if np.all(joints_vis[sk, 0] > 0):
            if np.all(joints[sk, 0] > 0):
                pt0 = (int(joints[sk[0]][0]), int(joints[sk[0]][1]))
                pt1 = (int(joints[sk[1]][0]), int(joints[sk[1]][1]))
                cv2.line(img=img_cv2, pt1=pt0, pt2=pt1, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
        for joint, joint_vis in zip(joints, joints_vis):
            if joint_vis[0]:
                cv2.circle(img_cv2, (int(joint[0]), int(joint[1])), 2, [255, 255, 0], 2)
        return img_cv2


if __name__ == '__main__':

    pose_infer = PoseInference()
    img_fpath = './test.png'
    img_cv2 = cv2.imread(img_fpath)
    # do predict
    predict = pose_infer.predict(img_cv2)

    # visualize
    draw = pose_infer.draw_img_with_joints(img_cv2, np.array(predict * 4), np.ones((predict.shape[0], 1)),)
    cv2.imwrite('result.jpg', draw)

