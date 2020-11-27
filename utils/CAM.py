# encoding: utf-8
import numpy as np
from os import listdir
import skimage.transform
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import os
import pickle
from collections import defaultdict
from collections import OrderedDict
from PIL import Image, ImageDraw
import PIL.ImageOps
import skimage
from skimage.io import *
from skimage.transform import *

import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.ndimage import binary_dilation
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from config import *

class CAM(object):
    
    def __init__(self, TRAN_SIZE, TRAN_CROP):
        self.TRAN_SIZE = TRAN_SIZE
        self.TRAN_CROP = TRAN_CROP

    # generate class activation mapping for the predicted classed
    def returnCAM(self, feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 224x224
        size_upsample = (self.TRAN_CROP, self.TRAN_CROP)
        bz, nc, h, w = feature_conv.shape

        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc,h*w)))
        #cam = weight_softmax[class_idx]*(feature_conv.reshape((nc,h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        cam_img = cv2.resize(cam_img, size_upsample)
        return cam_img

    def returnBox(self, data, width, height): #predicted bounding boxes
        # Find local maxima
        neighborhood_size = 100
        threshold = .1

        data_max = filters.maximum_filter(data, neighborhood_size)
        maxima = (data == data_max)
        data_min = filters.minimum_filter(data, neighborhood_size)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0
        for _ in range(5):
            maxima = binary_dilation(maxima) 
        labeled, num_objects = ndimage.label(maxima)
        #slices = ndimage.find_objects(labeled)
        xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))
        #get the hot points
        x_c, y_c, data_xy = 0, 0, 0.0
        for pt in xy:
            if data[int(pt[0]), int(pt[1])] > data_xy:
                data_xy = data[int(pt[0]), int(pt[1])]
                x_c = int(pt[0])
                y_c = int(pt[1]) 
        #resize the box to the size of orignal image
        x_scale = int(width/self.TRAN_SIZE)
        y_scale = int(height/self.TRAN_SIZE)
        crop_del = (self.TRAN_SIZE-self.TRAN_CROP)/2
        posX = (x_c+crop_del)*x_scale
        posY = (y_c+crop_del)*y_scale
        sizeX = config['sizeX']*x_scale
        sizeY = config['sizeY']*y_scale

        return [posX, posY, sizeX, sizeY]