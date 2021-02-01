import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
import time
import random
from sklearn.model_selection import train_test_split

from config import *

"""
Dataset: Diabetic Retinopathy Detection
https://www.kaggle.com/c/diabetic-retinopathy-detection/data
1) 35108 images
2）Label:0 - No DR, 1 - Mild, 2 - Moderate, 3 - Severe, 4 - Proliferative DR
"""


class DatasetGenerator(Dataset):
    def __init__(self, path_to_img_dir, path_to_dataset_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        for file_path in path_to_dataset_file:
            with open(file_path, "r") as f:
                for line in f:
                    items = line.split(',')
                    image_name = items[0] + '.jpeg'
                    label = items[1:]
                    label = [int(i) for i in label]
                    image_name = os.path.join(path_to_img_dir, image_name)
                    image_names.append(image_name)
                    labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

transform_seq_tr = transforms.Compose([
    transforms.Resize((config['TRAN_SIZE'], config['TRAN_SIZE'])),
    transforms.CenterCrop(config['TRAN_CROP']),
    transforms.ToTensor(),
    # normalize,
])

transform_seq_te = transforms.Compose([
    transforms.Resize((config['TRAN_SIZE'], config['TRAN_SIZE'])),
    transforms.CenterCrop(config['TRAN_CROP']),
    transforms.ToTensor(),
    # normalize,
])

PATH_TO_IMAGES_DIR = '/data/home/fangjiansheng/code/DR/dataset/images/'
PATH_TO_TRAIN_FILE = '/data/home/fangjiansheng/code/DR/Fundus2.0/datasets/train.txt'
PATH_TO_VAL_FILE = '/data/home/fangjiansheng/code/DR/Fundus2.0/datasets/val.txt'
PATH_TO_TEST_FILE = '/data/home/fangjiansheng/code/DR/Fundus2.0/datasets/test.txt'


def get_train_dataloader(batch_size, shuffle, num_workers):
    path_to_dataset_file = None
    if shuffle == True:  # for training
        path_to_dataset_file = [PATH_TO_TRAIN_FILE]
    else:  # for test
        path_to_dataset_file = [PATH_TO_TRAIN_FILE, PATH_TO_VAL_FILE]

    # dataset_train = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
    #                                  path_to_dataset_file=path_to_dataset_file, transform=transform_seq_tr)
    #
    dataset_train = Hard_Triplet(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                 path_to_dataset_file=path_to_dataset_file, transform=transform_seq_tr)

    # sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train) #for multi cpu and multi gpu
    # data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, sampler = sampler_train,
    # shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_train


def get_validation_dataloader(batch_size, shuffle, num_workers):
    # dataset_validation = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
    #                                       path_to_dataset_file=[PATH_TO_VAL_FILE], transform=transform_seq_te)
    #
    dataset_validation = Hard_Triplet(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                      path_to_dataset_file=[PATH_TO_VAL_FILE], transform=transform_seq_te)

    data_loader_validation = DataLoader(dataset=dataset_validation, batch_size=batch_size,
                                        shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_validation


def get_test_dataloader(batch_size, shuffle, num_workers):
    # dataset_test = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
    #                                 path_to_dataset_file=[PATH_TO_TEST_FILE], transform=transform_seq_te)
    dataset_test = Hard_Triplet(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                path_to_dataset_file=[PATH_TO_TEST_FILE], transform=transform_seq_te)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test


def splitKaggleDR(dataset_path):
    # case_id = list(set([ x.split('_')[0] for x in datas['image'] ])) #re-duplicate
    # ratio_test, ratio_val =  int(0.2*len(case_id)), int(0.1*len(case_id)) #trainset:valtest:testset=7:1:2
    # case_id_test = random.sample(case_id, ratio_test)
    # case_id = np.setdiff1d(np.array(case_id), np.array(case_id_test)) #Subtraction
    # case_id_val = random.sample(case_id.tolist(), ratio_val)
    # case_id_train = np.setdiff1d(np.array(case_id), np.array(case_id_val)) #Subtraction
    datas = pd.read_csv(dataset_path, sep=',')
    labels = pd.get_dummies(datas["level"])
    images = datas[['image']]
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.20, random_state=11)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=22)
    print("\r trainset shape: {}".format(y_train.shape))
    print("\r valset shape: {}".format(y_val.shape))
    print("\r testset shape: {}".format(y_test.shape))


def get_all_hard_triplets(path_to_img_dir, path_to_dataset_file):
    image_names = dict()
    for i in range(N_CLASSES):
        image_names[i] = []

    for file_path in path_to_dataset_file:
        with open(file_path, 'r') as f:
            for line in f:
                items = line.split(',')
                image_name = items[0] + '.jpeg'
                try:
                    label = items[1:].index('1')
                except:
                    label = items[1:].index('1\n')
                image_name = os.path.join(path_to_img_dir, image_name)
                image_names[label].append(image_name)

    triplets = []
    bit_clz = {0: [1, 0, 0, 0, 0], 1: [0, 1, 0, 0, 0], 2: [0, 0, 1, 0, 0], 3: [0, 0, 0, 1, 0], 4: [0, 0, 0, 0, 1]}
    for clz, img_list in image_names.items():
        for anchor in img_list:
            idx = random.randint(0, len(img_list) - 1)
            pos = img_list[idx]
            neg_clz = random.randint(0, N_CLASSES - 1)
            while neg_clz == clz:
                neg_clz = random.randint(0, N_CLASSES - 1)

            neg_list = image_names[neg_clz]
            idx = random.randint(0, len(neg_list) - 1)
            neg = neg_list[idx]
            # triplets.append((anchor,pos,neg,clz,clz,neg_clz))
            triplets.append((anchor, pos, neg, bit_clz[clz], bit_clz[clz], bit_clz[neg_clz]))
    return triplets


class Hard_Triplet(Dataset):
    '''
    基于FaceNet Hard Triplet
    '''

    def __init__(self, path_to_img_dir, path_to_dataset_file, transform):
        '''
        hard triplets初始化, 数据预处理方式
        '''
        self.triplets = get_all_hard_triplets(path_to_img_dir, path_to_dataset_file)

        # 数据预处理
        if transform == None:
            self.transform = transforms.Compose([
                transforms.Resize((config['TRAN_SIZE'], config['TRAN_SIZE'])),
                transforms.CenterCrop(config['TRAN_CROP']),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        else:
            self.transform = transform

    def __getitem__(self, index):
        '''
        一次获取一个hard triplet
        '''
        triplet = self.triplets[index]
        data = [self.transform(Image.open(img).convert('RGB')) for img in triplet[:3]]
        label = triplet[3:]
        return data, torch.FloatTensor(label)

    def __len__(self):
        '''
        返回所有hard triplet数量
        '''
        return len(self.triplets)


if __name__ == "__main__":
    # split trainset\valset\testset
    # dataset_path = '/data/pycode/FundusDR/datasets/trainLabels_cropped_fjs.csv'
    # splitKaggleDR(dataset_path)

    # for debug
    dataloader_train = get_train_dataloader(batch_size=32, shuffle=True, num_workers=8)
    for batch_idx, (image, label) in enumerate(dataloader_train):
        a = label
        break
