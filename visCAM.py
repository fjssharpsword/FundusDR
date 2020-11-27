# encoding: utf-8
"""
Training implementation
Author: Jason.Fang
Update time: 11/27/2020
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

#self-defined
from datasets.KaggleDR import get_train_dataloader, get_validation_dataloader, get_test_dataloader
from utils.Evaluation import compute_AUCs, compute_ROCCurve
from nets.ATNet import ATNet
from nets.ResNet import resnet50
from nets.DenseNet import DenseNet121
from config import *

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
np.set_printoptions(suppress=True) #to float

def LocHeatmap(CKPT_PATH, model_name):
    print('********************load data********************')
    dataloader_val = get_validation_dataloader(batch_size=1, shuffle=False, num_workers=0)
    image_names = dataloader_val.dataset.image_names
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if model_name == 'ATNet':
        model = ATNet(num_classes=N_CLASSES, is_pre_trained=True).cuda()#initialize model 
    elif model_name == 'ResNet':
        model = resnet50(t_num_classes=N_CLASSES, pretrained=True).cuda()#initialize model
    elif model_name == 'DenseNet':
        model = DenseNet121(num_classes=N_CLASSES, is_pre_trained=True).cuda()#initialize model
    else: 
        print('No required model')
        return #over

    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    if os.path.isfile(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> loaded model checkpoint: "+CKPT_PATH)
    print('******************** load model succeed!********************')

    print('******* begin visulization!*********')   
    model.eval()# switch to evaluate mode
    cls_weights = list(model.parameters())
    weight_softmax = np.squeeze(cls_weights[-6].data.cpu().numpy())#classes*1024
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(dataloader_val):
            var_image = torch.autograd.Variable(image).cuda()
            var_output = model(var_image)#forword
            h_x = F.softmax(var_output, dim=1).data.squeeze()#softmax
            probs, idx = h_x.sort(0, True) #probabilities of classe
            var_feature = model.dense_net_121.features(var_image) #get feature maps
            cam_img = returnCAM(var_feature.cpu().data.numpy(), weight_softmax, idx[0].item())
            box = returnBox(cam_img)
            #plot
            color_map = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
            image = image + 1 #[-1,1]->[1, 2]
            image = (image - image.min()) / (image.max() - image.min()) #[1, 2]->[0,1]
            image = image.numpy()*255 #[0,1] ->[0,255]
            #output_img = cv2.addWeighted(image, 0.7, color_map, 0.3, 0)
            fig, ax = plt.subplots(1)# Create figure and axes
            ax.imshow(image)
            ax.axis('off')
            fig.savefig(config['img_path']+str(batch_idx)+'.jpg')
            #sys.stdout.write('\r Visualization process: = {}'.format(batch_idx+1))
            #sys.stdout.flush()
            break

            """
            output_img = OverlapImages(gtbox[0].numpy(), predBoxes, heat_map, raw_img) #crop heatmap and merge
            
            x,y,w,h = gtbox[0][0], gtbox[0][1], gtbox[0][2], gtbox[0][3]
            rect = patches.Rectangle((x, y), w, h,linewidth=2, edgecolor='r', facecolor='none')# Create a Rectangle patch
            ax.add_patch(rect)# Add the patch to the Axes
            for i in range(len(predBoxes)):
                x_p,y_p,w_p,h_p = predBoxes[i][0], predBoxes[i][1], predBoxes[i][2], predBoxes[i][3]
                IoU_score, _ = compute_IoUs_and_Dices([x,y,w,h],[x_p,y_p,w_p,h_p])
                rect = patches.Rectangle((x_p, y_p), w_p, h_p,linewidth=2, edgecolor='b', facecolor='none')# Create a Rectangle patch
                ax.add_patch(rect)# Add the patch to the Axes
                ax.text(x_p, y_p, '{:.4f}'.format(IoU_score))
            ax.axis('off')

            gt_idx = np.where(label[0]==1)[0][0]
            gt_label = np.array(CLASS_NAMES)[gt_idx]
            gt_idx = np.where(idx.cpu().numpy()==gt_idx)[0][0]
            img_file = 'gt-{}-{:.4f}_pred-{}-{:.4f}'.format(gt_label, probs[gt_idx].cpu().item(), CLASS_NAMES[idx[0].item()], probs[0].cpu().item())
            fig.savefig('./Imgs/'+img_file+'.jpg')
            
            """

# generate class activation mapping for the predicted classed
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 224x224
    size_upsample = (config['TRAN_CROP'], config['TRAN_CROP'])
    bz, nc, h, w = feature_conv.shape

    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc,h*w)))
    #cam = weight_softmax[class_idx]*(feature_conv.reshape((nc,h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, size_upsample)
    return cam_img

def returnBox(data): #predicted bounding boxes
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
    return [x_c, y_c]

def main(model_name):
    CKPT_PATH = config['CKPT_PATH']+ model_name +'/best_model.pkl'
    LocHeatmap(CKPT_PATH, model_name) #for location

if __name__ == '__main__':

    main(model_name = 'ATNet')