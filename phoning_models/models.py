import os
import time
import torch
import torch.nn as nn
import torchvision.models as torch_models

from checkpoint.checkpoint import CheckpointMgr

class Phoning_Model_MLU(nn.Module):
    """
    RGB
    224x224
    """
    def __init__(self):
        super(Phoning_Model_MLU, self).__init__()
        self.backbone = torch_models.resnet34(pretrained=False)
        self.backbone.fc = nn.Linear(512,3)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, input):
        ###0 正常，1 打电话 2 玩手机####
        output = self.backbone(input) #[B,3]
        output = self.softmax(output) #[B,3]
        return output

    def load_checkpoint(self, pthfile):
        save_dict = torch.load(pthfile, map_location="cpu")
        save_dict = save_dict['state_dict'] if 'state_dict' in save_dict.keys() else save_dict
        """特殊处理, 针对多卡训练结果"""
        save_dict = {k.replace('module.',''):v for k,v in save_dict.items()}
        self.backbone.load_state_dict(save_dict, strict=True)
        return self

# def preprocess(image_path):
#     image = Image.open(image_path).convert("RGB")
#     image = np.asarray(image)
#     h,w,_ = image.shape
#     if h/w >=2:
#         image = image[0:int(h*0.5),:]
#     elif 1.5 <= h/w < 2:
#         image = image[0:int(h*0.8), :]
#     else:
#         image = image
#     # cv2.imwrite("/data/udisk3/data/0125/"+"crop_"+image_path.split("/")[-1],image)
#     image = Image.fromarray(image)
#
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#     trans = transforms.Compose([
#         transforms.Resize((256,256)),
#         transforms.ToTensor(),
#         normalize])
#     image = trans(image)
#     image = image.unsqueeze(0)
#     return image
#
# def test():
#     args = parser.parse_args()
#
#     model = models.resnet34(pretrained=False)
#     model.fc = nn.Linear(512,3)
#     model = model.cuda()
#
#     if os.path.isfile(args.weights):
#         print("=> loading checkpoint '{}'".format(args.weights))
#         load_checkpoint(model, args.weights)
#     else:
#         print("=> no checkpoint found at '{}'".format(args.weights))
#
#
#     cudnn.benchmark = True
#     model.eval()
#     image_root = Path(args.data)
#     if image_root.is_file():
#         image_lst = image_root.read_text().splitlines()
#         label_lst = list(map(lambda x:x.strip().split(" ")[-1],image_lst))
#         image_lst = list(map(lambda x:x.strip().split(" ")[0],image_lst))
#     else:
#         phone_folder = image_root
#         image_lst = list(phone_folder.glob("*.jpg"))
#
#     for image_file in tqdm(image_lst):
#         image = preprocess(str(image_file))
#         image = image.cuda()
#         output = model.forward(image)
#         logits = output.softmax(dim=1).detach().cpu().numpy()
#
#         ###0 正常，1 打电话 2 玩手机####