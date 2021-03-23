# encoding: utf-8
"""
Training implementation
Author: Jason.Fang
Update time: 16/03/2021
"""
import re
import sys
import os
import cv2
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
from skimage.measure import label
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.ndimage import binary_dilation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import PIL.ImageOps
import torchvision.transforms as transforms

#Visualize the mirror images
def genMirrorImages():
    """
    transform_tensor = transforms.Compose([transforms.ToTensor()]) #to tesnor [0,1]
    img_path = '/data/pycode/FundusDR/imgs/IDRiD_085.jpg'
    image = Image.open(img_path).convert('RGB')
    image = transform_tensor(image)
    avg_out = torch.mean(image, dim=0, keepdim=True)
    avg_img = transforms.ToPILImage()(avg_out*image).convert('RGB')
    avg_img.save('/data/pycode/FundusDR/imgs/IDRiD_085_avg.jpg')
    max_out, _ = torch.max(image, dim=0, keepdim=True)
    max_img = transforms.ToPILImage()(max_out*image).convert('RGB')
    max_img.save('/data/pycode/FundusDR/imgs/IDRiD_085_max.jpg')
    min_out, _ = torch.min(image, dim=0, keepdim=True)
    min_img = transforms.ToPILImage()(min_out*image).convert('RGB')
    min_img.save('/data/pycode/FundusDR/imgs/IDRiD_085_min.jpg')
    """
    """
    #https://zhuanlan.zhihu.com/p/74053773
    img_path = '/data/pycode/FundusDR/imgs/IDRiD_085.jpg'
    image = Image.open(img_path).convert('RGB')

    image_contour = image.filter(ImageFilter.SMOOTH)
    image_contour.save('/data/pycode/FundusDR/imgs/IDRiD_085_contour.jpg')
    """
    img_path = '/data/pycode/FundusDR/imgs/IDRiD_53.jpg'
    img = Image.open(img_path)
    img_horizontal = img.transpose(Image.FLIP_LEFT_RIGHT) #Flip left and right
    img_horizontal.save('/data/pycode/FundusDR/imgs/IDRiD_53_horizontal.jpg')
    img_vertical = img.transpose(Image.FLIP_TOP_BOTTOM) #Flip top and buttom
    img_vertical.save('/data/pycode/FundusDR/imgs/IDRiD_53_vertical.jpg')
    
def main():
    genMirrorImages()

if __name__ == '__main__':
    main()