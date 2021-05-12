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
from sklearn import preprocessing

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

PATH_TO_IMAGES_DIR_TRAIN = '/data/fjsdata/fundus/IDRID/ASegmentation/Images/TrainingSet/'
PATH_TO_IMAGES_DIR_TEST = '/data/fjsdata/fundus/IDRID/ASegmentation/Images/TestingSet/'
PATH_TO_FILES_DIR_TRAIN = '/data/pycode/Thesis/datasets/train.txt'
PATH_TO_FILES_DIR_TEST = '/data/pycode/Thesis/datasets/test.txt'
PATH_TO_MASKS_DIR_TRAIN = '/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TrainingSet/Haemorrhages/'
PATH_TO_MASKS_DIR_TEST = '/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TestingSet/Haemorrhages/'

def SegHE_Annotation():
    #number and size of hemorrhages
    #trianset 
    trIDs, trLbs = [], []
    for root, dirs, files in os.walk(PATH_TO_MASKS_DIR_TRAIN):
        for file in files:
            ID = os.path.splitext(file)[0]
            trIDs.append(ID[:-3])#'_HE' 
            file_path = os.path.join(PATH_TO_MASKS_DIR_TRAIN + file)
            img = cv2.imread(file_path, cv2.COLOR_BGR2GRAY) #binary image
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
            trLbs.append(num_labels)
   
    #trestset
    teIDs, teLbs = [], []
    for root, dirs, files in os.walk(PATH_TO_MASKS_DIR_TEST):
        for file in files:
            ID = os.path.splitext(file)[0]
            teIDs.append(ID[:-3])#'_HE' 
            file_path = os.path.join(PATH_TO_MASKS_DIR_TEST + file)
            img = cv2.imread(file_path, cv2.COLOR_BGR2GRAY) #binary image
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
            teLbs.append(num_labels)

    #min_max_scaler = preprocessing.MinMaxScaler()
    #min_max_scaler.fit(np.array(trLbs+teLbs)[:, np.newaxis])
    #trLbs = np.round(min_max_scaler.transform(np.array(trLbs)[:, np.newaxis]),2)
    #teLbs = np.round(min_max_scaler.transform(np.array(teLbs)[:, np.newaxis]),2)
    pd.concat([pd.DataFrame(trIDs), pd.DataFrame(trLbs)], axis=1).to_csv('/data/pycode/Thesis/datasets/train.txt', index=False, header=False, sep=',')
    pd.concat([pd.DataFrame(teIDs), pd.DataFrame(teLbs)], axis=1).to_csv('/data/pycode/Thesis/datasets/test.txt', index=False, header=False, sep=',')


class DatasetGenerator(Dataset):
    def __init__(self, path_to_img_dir, path_to_dataset_file):
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
                    image_name= items[0] + '.jpg'
                    label = [int(items[1])]#[eval(items[1])]#.replace('\n','')
                    image_name = os.path.join(path_to_img_dir, image_name)
                    image_names.append(image_name)
                    labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform_seq = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
            ])

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        image = self.transform_seq(image)
        label = self.labels[index]
        label = torch.as_tensor(label, dtype=torch.float32) 
  
        return image, label

    def __len__(self):
        return len(self.labels)

def get_train_dataloader(batch_size, shuffle, num_workers):
    dataset_train = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR_TRAIN, path_to_dataset_file=[PATH_TO_FILES_DIR_TRAIN])
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_train

def get_test_dataloader(batch_size, shuffle, num_workers):
    dataset_test = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR_TEST, path_to_dataset_file=[PATH_TO_FILES_DIR_TEST])
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test

if __name__ == "__main__":
    #for debug   
    #SegHE_Annotation()
    dataloader_train = get_train_dataloader(batch_size=10, shuffle=True, num_workers=0)
    for batch_idx, (image, label) in enumerate(dataloader_train):
        print(image.shape)
        print(label.shape)
        break
