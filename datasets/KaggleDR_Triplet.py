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
import sys
from sklearn.model_selection import train_test_split

#define by myself
sys.path.append("..") 
from FundusDR.config import *
#from config import *

"""
Dataset: Diabetic Retinopathy Detection
https://www.kaggle.com/c/diabetic-retinopathy-detection/data
1) 35108 images
2）Label:0 - No DR, 1 - Mild, 2 - Moderate, 3 - Severe, 4 - Proliferative DR
"""

def get_all_hard_triplets(path_to_img_dir, path_to_dataset_file):
    image_names = dict()
    for i in range(N_CLASSES):
        image_names[i] = []

    for file_path in path_to_dataset_file:
        with open(file_path, 'r') as f:
            for line in f:
                items = line.split(',')
                image_name = items[0] + '.jpeg'
                label = [int(i) for i in items[1:]].index(1)
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
    Ref: FaceNet Hard Triplet
    '''
    def __init__(self, path_to_img_dir, path_to_dataset_file, transform=None):
        '''
        hard triplets: sampling triplet images
        '''
        self.triplets = get_all_hard_triplets(path_to_img_dir, path_to_dataset_file)
        self.transform = transform #preprocessing

    def __getitem__(self, index):
        '''
        hard triplet: get
        '''
        triplet = self.triplets[index]
        data = [self.transform(Image.open(img).convert('RGB')) for img in triplet[:3]]
        label = [torch.FloatTensor(lbl) for lbl in triplet[3:]]
        return data, label

    def __len__(self):
        '''
        num of triplet-samples
        '''
        return len(self.triplets)

transform_seq = transforms.Compose([
    transforms.Resize((config['TRAN_SIZE'], config['TRAN_SIZE'])),
    transforms.ToTensor() #to tesnor [0,1]
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
])

PATH_TO_IMAGES_DIR = '/data/fjsdata/fundus/Fundus_DR_grading/images/resized_train_cropped/resized_train_cropped/'
PATH_TO_TRAIN_FILE = '/data/pycode/FundusDR/datasets/train.txt'
PATH_TO_VAL_FILE = '/data/pycode/FundusDR/datasets/val.txt'
PATH_TO_TEST_FILE = '/data/pycode/FundusDR/datasets/test.txt'

def get_train_dataloader(batch_size, shuffle, num_workers):
    dataset_train = Hard_Triplet(path_to_img_dir=PATH_TO_IMAGES_DIR,path_to_dataset_file=[PATH_TO_TRAIN_FILE,PATH_TO_VAL_FILE], transform=transform_seq)

    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_train


def get_validation_dataloader(batch_size, shuffle, num_workers):
    dataset_validation = Hard_Triplet(path_to_img_dir=PATH_TO_IMAGES_DIR,path_to_dataset_file=[PATH_TO_VAL_FILE], transform=transform_seq)

    data_loader_validation = DataLoader(dataset=dataset_validation, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_validation


def get_test_dataloader(batch_size, shuffle, num_workers):
    dataset_test = Hard_Triplet(path_to_img_dir=PATH_TO_IMAGES_DIR, path_to_dataset_file=[PATH_TO_TEST_FILE], transform=transform_seq)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test

if __name__ == "__main__":
    # for debug
    dataloader_train = get_train_dataloader(batch_size=32, shuffle=True, num_workers=8)
    for batch_idx, (image, label) in enumerate(dataloader_train):
        print(image[0].shape)
        print(label[0].shape)
        break
