import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
import time
import random
import sys
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import PIL.ImageOps
from sklearn.model_selection import train_test_split

#define by myself
sys.path.append("..") 
from FundusDR.config import *
#from config import *
"""
Dataset: Indian Diabetic Retinopathy Image Dataset (IDRiD)
https://idrid.grand-challenge.org/
Link to access dataset: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
Data Descriptor: http://www.mdpi.com/2306-5729/3/3/25
First Results and Analysis: https://doi.org/10.1016/j.media.2019.101561
A. Localization: center pixel-locations of optic disc and fovea center for all 516 images;
B. Disease Grading: 516 images, 413(80%)images for training, 103(20%) images for test.
1) DR (diabetic retinopathy) grading: 0-no apparent retinopathy, 1-mild NPDR, 2-moderate NPDR, 3-Severe NPDR, 4-PDR
2) Risk of DME (diabetic macular edema): 0-no apparent EX(s), 1-Presence of EX(s) outside the radius of one disc diameter form the macula center,
                                        2-Presence of EX(s) within the radius of one disc diameter form the macula center.
C. Segmentation: 
1) 81 DR images, 54 for training and 27 for test.
2) types: optic disc(OD), microaneurysms(MA), soft exudates(SE), hard exudates(EX), hemorrhages(HE).
"""

class DatasetGenerator(Dataset):
    def __init__(self, path_to_img_dir, path_to_msk_dir):
        """
        Args:
            path_to_img_dir: path to image directory.
            path_to_msk_dir: path to mask directory.
            transform: optional transform to be applied on a sample.
        """
        imageIDs, maskIDs_MA, maskIDs_HE = [], [], []
        for root, dirs, files in os.walk(path_to_img_dir):
            for file in files:
                ID = os.path.join(path_to_img_dir + file)
                imageIDs.append(ID)
                file = os.path.splitext(file)[0]
                ID = os.path.join(path_to_msk_dir + 'Microaneurysms/' + file+'_MA.tif')
                maskIDs_MA.append(ID)
                ID = os.path.join(path_to_msk_dir + 'Haemorrhages/' + file+'_HE.tif')
                maskIDs_HE.append(ID)

        self.imageIDs = imageIDs
        self.maskIDs_MA = maskIDs_MA
        self.maskIDs_HE = maskIDs_HE
        self.transform_seq_image = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
            ])
        self.transform_seq_mask = transforms.Compose([
            transforms.Resize((256,256)),
            ])

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image = self.imageIDs[index]
        mask_ma = self.maskIDs_MA[index]
        mask_he = self.maskIDs_HE[index] 

        if self.imageIDs[index] == '/data/fjsdata/fundus/IDRID/ASegmentation/Images/TrainingSet/IDRiD_46.jpg':
            #show 
            image = Image.open(image).convert('RGBA')
            mask_ma = Image.open(mask_ma).convert('RGBA')
            mask_ma = transparent_back(mask_ma, 'ma')
            overlay = Image.alpha_composite(image, mask_ma)
            mask_he = Image.open(mask_he).convert('RGBA')
            mask_he = transparent_back(mask_he, 'he')
            overlay = Image.alpha_composite(overlay, mask_he)
            plt.imshow(overlay)#cmap='gray'
            plt.axis('off')

            plt.savefig(config['img_path']+'IDRiD_xx_overlay.jpg')

        image = self.transform_seq_image(Image.open(image).convert('RGB'))
        mask_ma = torch.FloatTensor(np.array(self.transform_seq_mask(Image.open(mask_ma))))
        mask_he = torch.FloatTensor(np.array(self.transform_seq_mask(Image.open(mask_he))))
  
        return image, mask_ma, mask_he

    def __len__(self):
        return len(self.imageIDs)

PATH_TO_IMAGES_DIR_TRAIN = '/data/fjsdata/fundus/IDRID/ASegmentation/Images/TrainingSet/'
PATH_TO_IMAGES_DIR_TEST = '/data/fjsdata/fundus/IDRID/ASegmentation/Images/TestingSet/'
PATH_TO_MASKS_DIR_TRAIN = '/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TrainingSet/'
PATH_TO_MASKS_DIR_TEST = '/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TestingSet/'

def get_train_dataloader(batch_size, shuffle, num_workers):
    dataset_train = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR_TRAIN, path_to_msk_dir=PATH_TO_MASKS_DIR_TRAIN)
    #sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train) #for multi cpu and multi gpu
    #data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, sampler = sampler_train, 
                                   #shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_train

def get_test_dataloader(batch_size, shuffle, num_workers):
    dataset_test = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR_TEST, path_to_msk_dir=PATH_TO_MASKS_DIR_TEST)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test

def transparent_back(img, cls='he'):
    #img = img.convert('RGBA')
    L, H = img.size
    color_0 = img.getpixel((0,0)) #alpha channel: 0~255
    for h in range(H):
        for l in range(L):
            dot = (l,h)
            color_1 = img.getpixel(dot)
            if color_1 == color_0:
                color_1 = color_1[:-1] + (0,)
                img.putpixel(dot,color_1)
            else: 
                if cls=='ma':
                    color_1 = ( 0, 0, 255, 255) #turn to blue  and transparency 
                    img.putpixel(dot,color_1)
                else: #'he'
                    color_1 = ( 0 , 255, 0, 255) #turn to green  and transparency 
                    img.putpixel(dot,color_1)
    return img

if __name__ == "__main__":
    #for debug   
    dataloader_train = get_train_dataloader(batch_size=64, shuffle=True, num_workers=0)
    for batch_idx, (image, mask_ma, mask_he) in enumerate(dataloader_train):
        print(image.shape)
        print(mask_ma.shape)
        print(mask_he.shape)
        break
