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
    def __init__(self, path_to_img_dir, path_to_lbl_dir):
        """
        Args:
            path_to_img_dir: path to image directory.
            path_to_lbl_dir: path to label directory.
            transform: optional transform to be applied on a sample.
        """
        self.CLASS_NAMES = ['Normal', "Mild NPDR", 'Moderate NPDR', 'Severe NPDR', 'PDR']
        datas = pd.read_csv(path_to_lbl_dir, sep=',')
        datas = datas.values #dataframe -> numpy
        image_list = []
        label_list = []
        for data in datas:
            image = path_to_img_dir + data[0]+'.jpg'
            image_list.append(image)
            label = np.zeros(len(self.CLASS_NAMES)) #one-hot
            label[int(data[1])] = 1
            label_list.append(label)

        self.image_list = image_list
        self.label_list = label_list
        self.transform_seq = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image = self.image_list[index] 
        image = self.transform_seq(Image.open(image).convert('RGB'))
        label = self.label_list[index]
        label = torch.as_tensor(label, dtype=torch.float32)
  
        return image, label

    def __len__(self):
        return len(self.image_list)

PATH_TO_IMAGES_DIR_TRAIN = '/data/fjsdata/fundus/IDRID/BDiseaseGrading/OriginalImages/TrainingSet/'
PATH_TO_IMAGES_DIR_TEST = '/data/fjsdata/fundus/IDRID/BDiseaseGrading/OriginalImages/TestingSet/'
PATH_TO_LABELS_DIR_TRAIN = '/data/fjsdata/fundus/IDRID/BDiseaseGrading/Groundtruths/IDRiD_Grading_Training.csv'
PATH_TO_LABELS_DIR_TEST = '/data/fjsdata/fundus/IDRID/BDiseaseGrading/Groundtruths/IDRiD_Grading_Testing.csv'

def get_train_dataset_fundus():
    dataset_train = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR_TRAIN, path_to_lbl_dir=PATH_TO_LABELS_DIR_TRAIN)
    return dataset_train

def get_test_dataset_fundus():
    dataset_test = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR_TEST, path_to_lbl_dir=PATH_TO_LABELS_DIR_TEST)
    return dataset_test

def get_dataset_fundus():
    dataset_train = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR_TRAIN, path_to_lbl_dir=PATH_TO_LABELS_DIR_TRAIN)
    dataset_test = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR_TEST, path_to_lbl_dir=PATH_TO_LABELS_DIR_TEST)
    return dataset_train+dataset_test
