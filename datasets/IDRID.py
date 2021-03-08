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
Dataset: Indian Diabetic Retinopathy Image Dataset (IDRiD)
https://idrid.grand-challenge.org/
Link to access dataset: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
Data Descriptor: http://www.mdpi.com/2306-5729/3/3/25
First Results and Analysis: https://doi.org/10.1016/j.media.2019.101561
A. Localization: center pixel-locations of optic disc and fovea center for all 516 images;
B. Disease Grading: 516 images, 413(80%)images for training, 103(20%) images for test.
1) DR (diabetic retinopathy) grading:0(no apparent DR) to 4(severe DR).
2) Risk of DME (diabetic macular edema): 0(no DME) to 2(severe DME).
C. Segmentation: 
1) 81 DR images, 54 for training and 27 for test.
2) types: optic disc(OD), microaneurysms(MA), soft exudates(SE), hard exudates(EX), hemorrhages(HE).
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
   transforms.Resize((config['TRAN_SIZE'],config['TRAN_SIZE'])),
   transforms.CenterCrop(config['TRAN_CROP']),
   transforms.ToTensor(),
   #normalize,
])

transform_seq_te = transforms.Compose([
   transforms.Resize((config['TRAN_SIZE'],config['TRAN_SIZE'])),
   transforms.CenterCrop(config['TRAN_CROP']),
   transforms.ToTensor(),
   #normalize,
])

PATH_TO_IMAGES_DIR = '/data/fjsdata/fundus/Fundus_DR_grading/images/resized_train_cropped/resized_train_cropped/'
PATH_TO_TRAIN_FILE = '/data/pycode/FundusDR/datasets/train.txt'
PATH_TO_VAL_FILE = '/data/pycode/FundusDR/datasets/val.txt'
PATH_TO_TEST_FILE = '/data/pycode/FundusDR/datasets/test.txt'

def get_train_dataloader(batch_size, shuffle, num_workers):
    path_to_dataset_file = None
    if shuffle == True: #for training
        path_to_dataset_file = [PATH_TO_TRAIN_FILE]
    else: #for test
        path_to_dataset_file = [PATH_TO_TRAIN_FILE, PATH_TO_VAL_FILE]
        
    dataset_train = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                     path_to_dataset_file=path_to_dataset_file, transform=transform_seq_tr)
    #sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train) #for multi cpu and multi gpu
    #data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, sampler = sampler_train, 
                                   #shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_train

def get_validation_dataloader(batch_size, shuffle, num_workers):
    dataset_validation = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                          path_to_dataset_file=[PATH_TO_VAL_FILE], transform=transform_seq_te)
    data_loader_validation = DataLoader(dataset=dataset_validation, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_validation


def get_test_dataloader(batch_size, shuffle, num_workers):
    dataset_test = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                    path_to_dataset_file=[PATH_TO_TEST_FILE], transform=transform_seq_te)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
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
    dataloader_train = get_train_dataloader(batch_size=512, shuffle=True, num_workers=0)
    for batch_idx, (image, label) in enumerate(dataloader_train):
        print(label[0])
        break
