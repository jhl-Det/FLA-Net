import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Compose, Resize

from PIL import Image
import random
from torchvision.transforms import ToTensor as torchtotensor
import csv
import torch.utils.data as data
import numpy as np
################################iamge################################
class Compose_imglabel(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label


class Random_horizontal_flip(object):
    def _horizontal_flip(self, img, label):
        # dsa
        return img.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT)

    def __init__(self, prob):
        '''
        :param prob: should be (0,1)
        '''
        assert prob >= 0 and prob <= 1, "prob should be [0,1]"
        self.prob = prob

    def __call__(self, img, label):
        '''
        flip img and label simultaneously
        :param img:should be PIL image
        :param label:should be PIL image
        :return:
        '''
        assert isinstance(img, Image.Image), "should be PIL image"
        assert isinstance(label, Image.Image), "should be PIL image"
        if random.random() < self.prob:
            return self._horizontal_flip(img, label)
        else:
            return img, label


class Random_crop_Resize(object):
    def _randomCrop(self, img, label):
        width, height = img.size
        x, y = random.randint(0, self.crop_size), random.randint(0, self.crop_size)
        region = [x, y, width - x, height - y]
        img, label = img.crop(region), label.crop(region)
        img = img.resize((width, height), Image.BILINEAR)
        label = label.resize((width, height), Image.NEAREST)
        return img, label

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, label):
        assert img.size == label.size, "img should have the same shape as label"
        return self._randomCrop(img, label)


class Resize(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img, label):
        img = img.resize((self.width, self.height), Image.BILINEAR)
        label = label.resize((self.width, self.height), Image.NEAREST)
        return img, label


class Normalize(object):
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, img, label):
        for i in range(3):
            img[:, :, i] -= float(self.mean[i])
        for i in range(3):
            img[:, :, i] /= float(self.std[i])
        return img, label


class toTensor(object):
    def __init__(self):
        self.totensor = torchtotensor()

    def __call__(self, img, label):
        img, label = self.totensor(img), self.totensor(label).long()
        return img, label


################################video################################
class Random_crop_Resize_Video(object):
    def _randomCrop(self, img, label, x, y):
        width, height = img.size
        region = [x, y, width - x, height - y]
        img, label = img.crop(region), label.crop(region)
        img = img.resize((width, height), Image.BILINEAR)
        label = label.resize((width, height), Image.NEAREST)
        return img, label

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, imgs, labels):
        res_img = []
        res_label = []
        x, y = random.randint(0, self.crop_size), random.randint(0, self.crop_size)
        for img, label in zip(imgs, labels):
            img, label = self._randomCrop(img, label, x, y)
            res_img.append(img)
            res_label.append(label)
        return res_img, res_label


class Random_horizontal_flip_video(object):
    def _horizontal_flip(self, img, label):
        return img.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT)

    def __init__(self, prob):
        '''
        :param prob: should be (0,1)
        '''
        assert prob >= 0 and prob <= 1, "prob should be [0,1]"
        self.prob = prob

    def __call__(self, imgs, labels):
        '''
        flip img and label simultaneously
        :param img:should be PIL image
        :param label:should be PIL image
        :return:
        '''
        if random.random() < self.prob:
            res_img = []
            res_label = []
            for img, label in zip(imgs, labels):
                img, label = self._horizontal_flip(img, label)
                res_img.append(img)
                res_label.append(label)
            return res_img, res_label
        else:
            return imgs, labels


class Resize_video(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, imgs, labels):
        res_img = []
        res_label = []
        for img, label in zip(imgs, labels):
            res_img.append(img.resize((self.width, self.height), Image.BILINEAR))
            res_label.append(label.resize((self.width, self.height), Image.NEAREST))
        return res_img, res_label


class Normalize_video(object):
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, imgs, labels):
        res_img = []
        for img in imgs:
            for i in range(3):
                img[:, :, i] -= float(self.mean[i])
            for i in range(3):
                img[:, :, i] /= float(self.std[i])
            res_img.append(img)
        return res_img, labels


class toTensor_video(object):
    def __init__(self):
        self.totensor = torchtotensor()

    def __call__(self, imgs, labels):
        res_img = []
        res_label = []
        for img, label in zip(imgs, labels):
            img, label = self.totensor(img), self.totensor(label).long()
            res_img.append(img)
            res_label.append(label)
        return res_img, res_label


# finetune dataset
class VideoDataset(Dataset):
    def __init__(self, video_root, split_file, video_time_clips=5, transform=None, time_interval=1):
        super(VideoDataset, self).__init__()
        self.time_clips = video_time_clips
        self.video_train_list = []
        cls_list = []
        with open(os.path.join(video_root, split_file), encoding='utf-8-sig') as f:
            for row in csv.reader(f, skipinitialspace=True):
                cls_list.append(row[0])
        

        self.video_filelist = {}
        for cls in cls_list:
            self.video_filelist[cls] = []
            cls_img_path = os.path.join(video_root, "img_list", cls)
            cls_label_path = os.path.join(video_root, "mask_list", cls)
            tmp_list = os.listdir(cls_img_path)
            tmp_list.sort()
            for filename in tmp_list:
                self.video_filelist[cls].append((
                    os.path.join(cls_img_path, filename),
                    os.path.join(cls_label_path, filename)
                ))

        # ensemble
        for cls in cls_list:
            li = self.video_filelist[cls]
            for begin in range(1, len(li) - (self.time_clips - 1) * time_interval - 1):
                batch_clips = []
                for t in range(self.time_clips):
                    batch_clips.append(li[begin + time_interval * t])
                self.video_train_list.append(batch_clips)
        self.img_label_transform = transform

    def __getitem__(self, idx):
        img_label_li = self.video_train_list[idx]
        IMG = None
        LABEL = None
        img_li = []
        label_li = []
        for idx, (img_path, label_path) in enumerate(img_label_li):
            img = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            img_li.append(img)
            label_li.append(label)
        img_li, label_li = self.img_label_transform(img_li, label_li)
        for idx, (img, label) in enumerate(zip(img_li, label_li)):
            if IMG is not None:
                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label
            else:
                IMG = torch.zeros(len(img_li), *(img.shape))
                LABEL = torch.zeros(len(img_li), *(label.shape))
                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label
        return IMG, LABEL

    def __len__(self):
        return len(self.video_train_list)


def get_video_dataset(frm_size, video_root, batchsize = 1, video_time_clips=5,split_file="train_list.csv", shuffle=True, num_workers=12, pin_memory=False):
    trsf_main = Compose_imglabel([
        Resize_video(frm_size[0], frm_size[1]),
        Random_crop_Resize_Video(7),
        Random_horizontal_flip_video(0.5),
        toTensor_video(),
        Normalize_video([0.4732661, 0.44874457, 0.3948762],
                      [0.22674961, 0.22012031, 0.2238305])
    ])
    train_loader = VideoDataset(video_root, split_file, video_time_clips, transform=trsf_main, time_interval=1)


    data_loader = data.DataLoader(dataset=train_loader,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    return data_loader


#test dataset and loader
class video_test_dataset:
    def __init__(self, video_root, split_file, testsize, time_clips=5):
        time_interval = 1
        self.time_clips = time_clips
        self.video_test_list = []
        cls_list = []
        with open(os.path.join(video_root, split_file), encoding='utf-8-sig') as f:
            for row in csv.reader(f, skipinitialspace=True):
                cls_list.append(row[0])
        

        self.video_filelist = {}
        for cls in cls_list:
            self.video_filelist[cls] = []
            cls_img_path = os.path.join(video_root, "img_list", cls)
            cls_label_path = os.path.join(video_root, "mask_list", cls)
            tmp_list = os.listdir(cls_img_path)
            tmp_list.sort()
            for filename in tmp_list:
                self.video_filelist[cls].append((
                    os.path.join(cls_img_path, filename),
                    os.path.join(cls_label_path, filename)
                ))

        # ensemble
        for cls in cls_list:
            li = self.video_filelist[cls]
            # print(li)
            begin = 0
            while begin < len(li) - 1:
                if len(li) - 1 - begin <= self.time_clips:
                    begin = len(li) - self.time_clips
                batch_clips = []
                for t in range(self.time_clips):
                    batch_clips.append(li[begin + time_interval * t])
                begin += self.time_clips
                self.video_test_list.append(batch_clips)
        # print(self.video_test_list)
        self.testsize = testsize
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.video_test_list)
        self.index = 0
        # exit()

    def load_data(self):
        img_path_li = self.video_test_list[self.index]
        # print("img_path_li", img_path_li)
        IMG = None
        LABEL = None
        img_li = []
        label_li = []
        names = []
        for idx, (img_path, label_path) in enumerate(img_path_li):
            names.append(img_path)
            img = self.rgb_loader(img_path)
            label = self.binary_loader(label_path)

            img_li.append(self.transform(img))
            label_li.append(self.gt_transform(label))
        # exit()
        for idx, (img, label) in enumerate(zip(img_li, label_li)):
            if IMG is not None:
                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label
            else:
                IMG = torch.zeros(len(img_li), *(img.shape))
                LABEL = torch.zeros(len(img_li), *(label.shape))
                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label


        self.index += 1
        self.index = self.index % self.size
        
        # print(names)
        return IMG, LABEL, names, None

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

