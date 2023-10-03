import torch
import torch.nn.functional as F
import sys
import torch.nn as nn
import numpy as np
import os, argparse
import cv2
from Code.lib.model import SPNet
from Code.utils.data import video_test_dataset
import segmentation_models_pytorch as smp
from tqdm import tqdm
from Code.lib.PNS_Network import PNSNet

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--gpu_id',   type=str, default='0', help='select gpu id')
parser.add_argument('--root',   type=str, default='/home/zl/Workspace/video_object_seg/GSFM/data/breast_lesion_seg', help='select gpu id')
parser.add_argument('--split_file',type=str, default='test_list.csv',help='test dataset path')
parser.add_argument('--ckpt',type=str,help='test dataset path', required=True)
parser.add_argument('--save_path',type=str,help='test dataset path', default="./inf_results")


opt = parser.parse_args()


#set device for test
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU ', opt.gpu_id)

 

model = PNSNet()

model.cuda()
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(opt.ckpt))
model = model.module
model.eval()

#test

save_path = os.path.join(opt.save_path, opt.ckpt.split("/")[-1].split(".")[0])
if not os.path.exists(save_path):
    os.makedirs(save_path)

test_loader  = video_test_dataset(opt.root, opt.split_file, opt.testsize)

for i in tqdm(range(test_loader.size)):
    image, gt, name, image_for_post = test_loader.load_data()
    name = name.split("/")[-2:]
    gt      = np.asarray(gt, np.float32)
    gt     /= (gt.max() + 1e-8)
    image   = image.cuda()
    res = model(image)
    res     = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
    res     = res.sigmoid().data.cpu().numpy().squeeze()
    res     = (res - res.min()) / (res.max() - res.min() + 1e-8)
    
    save_video_path = os.path.join(save_path, name[0])
    if not os.path.exists(save_video_path):
        os.makedirs(save_video_path)
    
    # print('save img to: ', os.path.join(save_video_path,name[-1]))
    cv2.imwrite(os.path.join(save_video_path,name[-1]),res*255)
print('Test Done!')
