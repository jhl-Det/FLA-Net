import os
import torch
import cv2
import torch.nn.functional as F
import numpy as np
from Code.utils.video_dataloader3 import test_dataset
import torch.backends.cudnn as cudnn
from Code.utils.options import opt
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU ', opt.gpu_id)

from segmentation_models_pytorch import create_model
  
cudnn.benchmark = True

model = create_model(
    arch=opt.arch, 
    encoder_name="timm-res2net50_26w_4s",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,)


num_gpus = len(opt.gpu_id.split(","))
if num_gpus > 1:
    model = torch.nn.DataParallel(model)
    opt.batchsize *= num_gpus

if(opt.load is not None):
    model.load_state_dict(torch.load(opt.load))
    print('load model from ',opt.load)

model.cuda()

#set the path
data_root = opt.data_root
val_file      = "test_list.csv"
save_path        = opt.save_path


if not os.path.exists(save_path):
    os.makedirs(save_path)

#load data
print('load data...')
test_loader  = test_dataset(data_root, val_file, opt.trainsize)
        
#test function
def val(test_loader,model,epoch,save_path):
    model.eval()
    with torch.no_grad():
        mae_sum=0
        for i in tqdm(range(test_loader.size)):
            image, gt, name, image_for_post, hm = test_loader.load_data()
            name = name.split("/")[-2:]
            gt      = np.asarray(gt, np.float32)
            gt     /= (gt.max() + 1e-8)
            image   = image.cuda()
            res, *_     = model(image)
            res     = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res     = res.sigmoid().data.cpu().numpy().squeeze()
            res     = (res - res.min()) / (res.max() - res.min() + 1e-8)

            res     = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res-gt))*1.0/(gt.shape[0]*gt.shape[1])

            save_video_path = os.path.join(save_path, name[0])
            if not os.path.exists(save_video_path):
                os.makedirs(save_video_path)
            
            # print('save img to: ', os.path.join(save_video_path,name[-1]))
            cv2.imwrite(os.path.join(save_video_path,name[-1]), res*255)
        mae = mae_sum/test_loader.size
        print(f"MAE: {round(mae, 3)}")
        print('Test Done!')
            
 
if __name__ == '__main__':
    print("Start eval...")    
    val(test_loader, model, 1, save_path)
