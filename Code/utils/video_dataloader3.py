import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
import csv
import torch
import json
#several data augumentation strategies
def cv_random_flip(img, label):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    #left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    #top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label
def randomCrop(image, label):
    border=30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)
def randomRotation(image,label):
    mode=Image.BICUBIC
    if random.random()>0.8:
        random_angle = np.random.randint(-15, 15)
        image=image.rotate(random_angle, mode)
        label=label.rotate(random_angle, mode)
    return image,label
def colorEnhance(image):
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image
def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im
    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))
def randomPeper(img):

    img=np.array(img)
    noiseNum=int(0.0015*img.shape[0]*img.shape[1])
    for i in range(noiseNum):

        randX=random.randint(0,img.shape[0]-1)  

        randY=random.randint(0,img.shape[1]-1)  

        if random.randint(0,1)==0:  

            img[randX,randY]=0  

        else:  

            img[randX,randY]=255 
    return Image.fromarray(img)  

# dataset for training
#The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
#(e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class HeatmapGenerator():
    def __init__(self, output_res, num_joints, sigma=-1):
        self.output_res = output_res
        self.num_joints = num_joints
        if sigma < 0:
            sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, joints):
        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        sigma = self.sigma
        for idx, pt in enumerate(joints):
            if pt[2] > 0:
                x, y = int(pt[0]), int(pt[1])
                if x < 0 or y < 0 or \
                    x >= self.output_res or y >= self.output_res:
                    continue

                ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                hms[idx, aa:bb, cc:dd] = np.maximum(
                    hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms

class SalObjDataset(data.Dataset):
    def __init__(self, root, split_file, trainsize, clip_len=3):
        self.trainsize = trainsize
        image_root = os.path.join(root, "img_list")
        vid_list = []
        with open(os.path.join(root, split_file), encoding='utf-8-sig') as f:
            for row in csv.reader(f, skipinitialspace=True):
                vid_list.append(row[0])
        
        self.images = []
        self.gts = []
        for vid in vid_list:
            vid_path = os.path.join(image_root, vid)
            frms = sorted(os.listdir(vid_path))
            for idx in range(len(frms)):
                clip = []
                for ii in range(clip_len):
                    pick_idx = idx + ii if idx - ii < 0 else idx - ii
                    if pick_idx >= len(frms):
                        pick_idx = - 1
                    clip.append(os.path.join(vid_path, frms[pick_idx]))
                self.images.append(clip)
                self.gts.append([x.replace("img_list", "mask_list") for x in clip])

        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        
        with open(os.path.join(root, "bbox.json"), encoding='utf-8') as a:
            self.bboxs = json.load(a)
        self.getHeatmap = HeatmapGenerator(self.trainsize, 1, 10)
        self.hm_transform = transforms.ToTensor()
    
    def __getitem__(self, index):
        images = [self.rgb_loader(x) for x in self.images[index]]
        gt = self.binary_loader(self.gts[index][0])
        # getHeatmap
        key_name = self.images[index][0].split("/")[-2:]

        bbox = self.bboxs[key_name[-2]][key_name[-1]]
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        center_x, center_y = bbox[0] + w//2, bbox[1] + h//2
        ratio = gt.size[0] / self.trainsize, gt.size[1] / self.trainsize
        center_x, center_y = int(center_x/ratio[0]), int(center_y/ratio[1])
        
        hm = self.getHeatmap([[center_x, center_y, 1]])
        hm = self.hm_transform(hm)


        gt = self.gt_transform(randomPeper(gt))
        images = [self.img_transform(x) for x in images]
        
        return torch.stack(images), gt, hm.permute(1,0,2)

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


###############################################################################
# 0919
#

#dataloader for training
def get_loader(image_root, split_file, batchsize, trainsize, clip_len=3, shuffle=True, num_workers=12, pin_memory=True):

    dataset = SalObjDataset(image_root, split_file, trainsize, clip_len)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader




#test dataset and loader
class test_dataset:
    def __init__(self, root, val_file, testsize, clip_len=3):
        self.testsize = testsize
        self.images = []
        image_root = os.path.join(root, "img_list")
        gt_root = os.path.join(root, "mask_list")
        vid_list = []
        with open(os.path.join(root, val_file), encoding='utf-8-sig') as f:
            for row in csv.reader(f, skipinitialspace=True):
                vid_list.append(row[0])
        self.images = []
        self.gts = []
        for vid in vid_list:
            vid_path = os.path.join(image_root, vid)
            frms = sorted(os.listdir(vid_path))
            for idx in range(len(frms)):
                clip = []
                for ii in range(clip_len):
                    
                    pick_idx = idx + ii if idx - ii < 0 else idx - ii
                    if pick_idx >= len(frms):
                        pick_idx = - 1
                    clip.append(os.path.join(vid_path, frms[pick_idx]))
                self.images.append(clip)
                self.gts.append([x.replace("img_list", "mask_list") for x in clip])


        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0
        with open(os.path.join(root, "bbox.json"), encoding='utf-8') as a:
            self.bboxs = json.load(a)
        self.getHeatmap = HeatmapGenerator(self.testsize, 1, 10)
        self.hm_transform = transforms.ToTensor()

    def load_data(self):
        images = [self.rgb_loader(x) for x in self.images[self.index]]
        images = [self.transform(x) for x in images]
        gt = self.binary_loader(self.gts[self.index][0])
        # getHeatmap
        key_name = self.images[self.index][0].split("/")[-2:]
        bbox = self.bboxs[key_name[-2]][key_name[-1]]
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        center_x, center_y = bbox[0] + w//2, bbox[1] + h//2
        ratio = gt.size[0] / self.testsize, gt.size[1] / self.testsize
        center_x, center_y = int(center_x/ratio[0]), int(center_y/ratio[1])
        
        hm = self.getHeatmap([[center_x, center_y, 1]])
        hm = self.hm_transform(hm)


        name = self.images[self.index][0]
        image_for_post=self.rgb_loader(self.images[self.index][0])
        image_for_post=image_for_post.resize(gt.size)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return torch.stack(images).unsqueeze(0), gt, name, np.array(image_for_post), hm

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def __len__(self):
        return self.size