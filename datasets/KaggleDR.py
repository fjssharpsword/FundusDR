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
2）Label:0 - No DR, 1 - Mild DR, 2 - Moderate DR, 3 - Severe DR, 4 - Proliferative DR
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
                    image_name= items[0] + '.jpeg'
                    label = items[1:]
                    label = [int(i) for i in label]
                    image_name = os.path.join(path_to_img_dir, image_name)
                    image_names.append(image_name)
                    labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform
        """
        #statistics of dataset
        labels_np = np.array(labels)
        multi_dis_num = 0
        for i in range(len(CLASS_NAMES)):
            num = len(np.where(labels_np[:,i]==1)[0])
            multi_dis_num = multi_dis_num + num
            print('Number of {} is {}'.format(CLASS_NAMES[i], num))
        print('Number of Multi Finding is {}'.format(multi_dis_num))
        """

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
        #if self.transform is not None:
            #image = self.transform(image)
        return self.transform(image), transform_flip_H(image), transform_flip_V(image), torch.FloatTensor(label) #for manet
        #return self.transform(image), torch.FloatTensor(label)  #for sanet 
        #return self.transform(image), torch.FloatTensor(label), image_name #for query


    def __len__(self):
        return len(self.image_names)

transform_seq = transforms.Compose([
    transforms.Resize((config['TRAN_SIZE'], config['TRAN_SIZE'])),
    transforms.ToTensor() #to tesnor [0,1]
])

transform_flip_H = transforms.Compose([
    transforms.Resize((config['TRAN_SIZE'], config['TRAN_SIZE'])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor() #to tesnor [0,1]
])
transform_flip_V = transforms.Compose([
    transforms.Resize((config['TRAN_SIZE'], config['TRAN_SIZE'])),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor() #to tesnor [0,1]
])

PATH_TO_IMAGES_DIR = '/data/fjsdata/fundus/Fundus_DR_grading/images/resized_train_cropped/resized_train_cropped/'
PATH_TO_TRAIN_FILE = '/data/pycode/FundusDR/datasets/train.txt'
PATH_TO_VAL_FILE = '/data/pycode/FundusDR/datasets/val.txt'
PATH_TO_TEST_FILE = '/data/pycode/FundusDR/datasets/test.txt'

def get_train_dataloader(batch_size, shuffle, num_workers):
    dataset_train = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,path_to_dataset_file=[PATH_TO_TRAIN_FILE,PATH_TO_VAL_FILE], transform=transform_seq)
    #sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train) #for multi cpu and multi gpu
    #data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, sampler = sampler_train, 
                                   #shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_train

def get_validation_dataloader(batch_size, shuffle, num_workers):
    dataset_validation = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR, path_to_dataset_file=[PATH_TO_VAL_FILE], transform=transform_seq)
    data_loader_validation = DataLoader(dataset=dataset_validation, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_validation


def get_test_dataloader(batch_size, shuffle, num_workers):
    dataset_test = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,path_to_dataset_file=[PATH_TO_TEST_FILE], transform=transform_seq)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test


def splitKaggleDR(dataset_path):
    #case_id = list(set([ x.split('_')[0] for x in datas['image'] ])) #re-duplicate 
    #ratio_test, ratio_val =  int(0.2*len(case_id)), int(0.1*len(case_id)) #trainset:valtest:testset=7:1:2
    #case_id_test = random.sample(case_id, ratio_test)
    #case_id = np.setdiff1d(np.array(case_id), np.array(case_id_test)) #Subtraction
    #case_id_val = random.sample(case_id.tolist(), ratio_val)
    #case_id_train = np.setdiff1d(np.array(case_id), np.array(case_id_val)) #Subtraction
    datas = pd.read_csv(dataset_path, sep=',')
    labels = pd.get_dummies(datas["level"])
    images = datas[['image']]
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.20, random_state=11)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=22)
    print("\r trainset shape: {}".format(y_train.shape)) 
    print("\r valset shape: {}".format(y_val.shape)) 
    print("\r testset shape: {}".format(y_test.shape)) 
    trainset = pd.concat([X_train, y_train], axis=1).to_csv('/data/pycode/FundusDR/datasets/train.txt', index=False, header=False, sep=',')
    valset = pd.concat([X_val, y_val], axis=1).to_csv('/data/pycode/FundusDR/datasets/val.txt', index=False, header=False, sep=',')
    testset = pd.concat([X_test, y_test], axis=1).to_csv('/data/pycode/FundusDR/datasets/test.txt', index=False, header=False, sep=',')
    
if __name__ == "__main__":
    #split trainset\valset\testset
    #dataset_path = '/data/pycode/FundusDR/datasets/trainLabels_cropped_fjs.csv'
    #splitKaggleDR(dataset_path)

    #for debug   
    dataloader_train = get_test_dataloader(batch_size=10, shuffle=True, num_workers=0)
    for batch_idx, (image, image_h, image_v, label) in enumerate(dataloader_train):
        print(image.shape)
        print(image_h.shape)
        print(image_v.shape)
        print(label.shape)
        break
