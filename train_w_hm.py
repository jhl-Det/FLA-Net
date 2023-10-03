import os
import torch
import torch.nn as nn

import torch.nn.functional as F
import sys
import numpy as np
from datetime import datetime
from Code.utils.video_dataloader3 import get_loader,test_dataset
from Code.utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from Code.utils.options import opt
from tqdm import tqdm
import segmentation_models_pytorch as smp

#set the device for training
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU ', opt.gpu_id)

from segmentation_models_pytorch import create_model
  
cudnn.benchmark = True

# build the model
# archs = [
#     Unet,
#     UnetPlusPlus,
#     MAnet,
#     Linknet,
#     FPN,
#     PSPNet,
#     DeepLabV3,
#     DeepLabV3Plus,
#     PAN,
# ]
# model = smp.Unet(
#     encoder_name="resnet50",
#     encoder_weights="imagenet",
#     in_channels=3,
#     classes=1,
# )
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
params    = model.parameters()
if opt.sgd:
    print("using sgd to finetune!")
    optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.9)
else:
    optimizer = torch.optim.Adam(params, opt.lr)


#set the path
data_root = opt.data_root
train_file    = "train_list.csv"
val_file      = "val_list.csv"

save_path        = opt.save_path


if not os.path.exists(save_path):
    os.makedirs(save_path)

#load data
print('load data...')
train_loader = get_loader(data_root, train_file, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader  = test_dataset(data_root, val_file, opt.trainsize)
total_step   = len(train_loader)


logging.basicConfig(filename=save_path+'log.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("BBSNet_unif-Train")
logging.info("Config")
logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(opt.epoch,opt.lr,opt.batchsize,opt.trainsize,opt.clip,opt.decay_rate,opt.load,save_path,opt.decay_epoch))

#set loss function
step = 0
writer     = SummaryWriter(save_path+'summary')
best_mae   = 1
best_epoch = 0

print(len(train_loader))

def heatmap_loss(pred, gt):
    # print(pred.shape, mask.shape)

    assert pred.size() == gt.size()
    loss = ((pred - gt)**2).expand_as(pred)
    loss = loss.mean(dim=2).mean(dim=1).mean(dim=0)
    return loss

def structure_loss(pred, mask):
    # print(pred.shape, mask.shape)
    # return CE2(pred, mask)

    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()



def train(train_loader, model, optimizer, epoch,save_path):
    global step
    model.train()
    loss_all=0
    epoch_step=0
    try:
        for i, (images, gts, hms) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            
            images   = images.cuda()
            gts      = gts.cuda()
            hms      = hms.cuda()


            ##
            pre_res, hm_preds  = model(images)
            
            loss    = structure_loss(pre_res, gts) 
            hm_loss = heatmap_loss(hm_preds[-1].squeeze(),  hms.squeeze())
            
            all_loss = loss + hm_loss
            all_loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step+=1
            epoch_step+=1
            loss_all+=loss.data
            if i % 50 == 0 or i == total_step or i==1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                    format( epoch, opt.epoch, i, total_step, loss))
                
        loss_all/=epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format( epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path+'epoch_{}.pth'.format(epoch))
            
    except KeyboardInterrupt: 
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path+'HyperNet_epoch_{}.pth'.format(epoch+1))
        print('save checkpoints successfully!')
        raise
        
        
        
#test function
def val(test_loader,model,epoch,save_path):
    global best_mae,best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum=0
        for i in tqdm(range(test_loader.size)):
            image, gt, name, img_for_post, hm = test_loader.load_data()
            gt      = np.asarray(gt, np.float32)
            gt     /= (gt.max() + 1e-8)
            image   = image.cuda()
            pre_res, _ = model(image)
            res     = pre_res
            res     = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res     = res.sigmoid().data.cpu().numpy().squeeze()
            res     = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res-gt))*1.0/(gt.shape[0]*gt.shape[1])
            
        mae = mae_sum/test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch,mae,best_mae,best_epoch))
        if epoch==1:
            best_mae = mae
        else:
            if mae<best_mae:
                best_mae   = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path+'epoch_best.pth')
                print('best epoch:{}'.format(epoch))
                
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch,mae,best_epoch,best_mae))
 
if __name__ == '__main__':
    print("Start train...")
    
    if opt.load is not None:
        val(test_loader, model, 1, save_path)

    for epoch in range(1, opt.epoch):
        
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        # val(test_loader, model, epoch, save_path)
        
        train(train_loader, model, optimizer, epoch, save_path)
        print("----"*10)
        #test
        if epoch % 5 == 0:
            val(test_loader, model, epoch, save_path)
