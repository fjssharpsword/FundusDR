# encoding: utf-8
"""
Training implementation for Mirror Attention
Author: Jason.Fang
Update time: 06/09/2021
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
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
from skimage.measure import label
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
from pyheatmap.heatmap import HeatMap
#self-defined
from config import *
from utils.logger import get_logger
from nets.SANet import SANet

#command parameters
parser = argparse.ArgumentParser(description='For FundusDR')
parser.add_argument('--model', type=str, default='SANet', help='SANet')
args = parser.parse_args()

#config
os.environ['CUDA_VISIBLE_DEVICES'] = config['CUDA_VISIBLE_DEVICES']
logger = get_logger(config['log_path'])


def vis_cls_map():
    print('********************load model********************')
    if args.model == 'SANet':
        model = SANet(num_classes=N_CLASSES).cuda()
        CKPT_PATH = config['CKPT_PATH'] + args.model + '_best_model_v1.0.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model.load_state_dict(checkpoint) #strict=False
            print("=> Loaded well-trained checkpoint from: "+CKPT_PATH)
    else: 
        print('No required model')
        return #over
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    model.eval()#turn to test mode
    print('******************** load model succeed!********************')

    query_img_path = '/data/fjsdata/fundus/IDRID/BDiseaseGrading/OriginalImages/TrainingSet/IDRiD_245.jpg' #
    transform_seq = transforms.Compose([
        transforms.Resize((config['TRAN_SIZE'], config['TRAN_SIZE'])),
        transforms.ToTensor() #to tesnor [0,1]
    ])
    query_img = Image.open(query_img_path).convert('RGB')

    #infer feature map
    var_image = torch.autograd.Variable(torch.unsqueeze(transform_seq(query_img),0)).cuda()
    x, out, map_bb, map_sa = model(var_image)
    heat_map = map_bb.data.cpu().squeeze(0).sum(dim=0) #1024*8*8
    heat_map = heat_map.squeeze(0).numpy()#.permute(1, 2, 0)
    heat_map = heat_map - heat_map.min()
    heat_map = heat_map / heat_map.max()

    #plot feature map
    fm_img =  np.asarray(query_img)
    width, height = fm_img.shape[0],fm_img.shape[1]
    heat_map = cv2.resize(heat_map,(height, width))
    heat_map = np.uint8(heat_map * 255.0)
    #heat_map = cv2.cvtColor(np.asarray(heat_map),cv2.COLOR_RGB2BGR)
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET) 
    output_img = cv2.addWeighted(fm_img, 0.7, heat_map, 0.3, 0)
    plt.imshow(output_img)
    plt.axis('off')
    plt.savefig('/data/pycode/FundusDR/imgs/IDRiD_245_moderateDR.jpg', dpi = 400)


def vis_seg_map():
    print('********************load model********************')
    if args.model == 'SANet':
        model = SANet(num_classes=N_CLASSES).cuda()
        CKPT_PATH = config['CKPT_PATH'] + args.model + '_best_model_v1.0.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model.load_state_dict(checkpoint) #strict=False
            print("=> Loaded well-trained checkpoint from: "+CKPT_PATH)
    else: 
        print('No required model')
        return #over
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    model.eval()#turn to test mode
    print('******************** load model succeed!********************')

    query_img_path = '/data/pycode/FundusDR/imgs/IDRiD_46.jpg' #IDRiD_53.jpg, IDRiD_09.jpg, IDRiD_46.jpg
    transform_seq = transforms.Compose([
        transforms.Resize((config['TRAN_SIZE'], config['TRAN_SIZE'])),
        transforms.ToTensor() #to tesnor [0,1]
    ])
    query_img = Image.open(query_img_path).convert('RGB')

    #infer feature map
    var_image = torch.autograd.Variable(torch.unsqueeze(transform_seq(query_img),0)).cuda()
    x, out, map_bb, map_sa = model(var_image)
    heat_map = map_bb.data.cpu().squeeze(0).sum(dim=0) #1024*8*8
    heat_map = heat_map.squeeze(0).numpy()#.permute(1, 2, 0)
    heat_map = heat_map - heat_map.min()
    heat_map = heat_map / heat_map.max()

    """
    #plot feature map
    fm_img =  np.asarray(query_img)
    width, height = fm_img.shape[0],fm_img.shape[1]
    heat_map = cv2.resize(heat_map,(height, width))
    heat_map = np.uint8(heat_map * 255.0)
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET) 
    output_img = cv2.addWeighted(fm_img, 0.7, heat_map, 0.3, 0)
    plt.imshow(output_img)
    plt.axis('off')
    plt.savefig('/data/pycode/FundusDR/imgs/IDRiD_53_map_sa_22.jpg', dpi = 400)
    """
    
    #plot heat map
    """
    sns_temp = sns.heatmap(heat_map, cmap="RdBu", annot=True) #hot_r #OrRd #RdBu
    sns_temp = sns_temp.get_figure()
    sns_temp.savefig('/data/pycode/FundusDR/imgs/IDRiD_53_sns_22.jpg', dpi = 400)
    """
   
    #plot lesion map
    fm_img =  np.asarray(query_img.resize((512,512)))
    heat_map = cv2.resize(heat_map,(512,512))

    mask_ma = os.path.join('/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TrainingSet/Microaneurysms/IDRiD_46_MA.tif')
    mask_ma = Image.open(mask_ma).convert('L').resize((512,512)) #0,76
    mask_he = os.path.join('/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TrainingSet/Haemorrhages/IDRiD_46_HE.tif')
    mask_he = Image.open(mask_he).convert('L').resize((512,512))
    mask = np.array(mask_he) + np.array(mask_ma)
    idx_mask = np.argwhere(mask>0) #return nonzero index
    hm_cord = []
    for xy in idx_mask:
        hm_cord.append([xy[0], xy[1], heat_map[xy[0], xy[1]]])
    
    hm_bg = Image.new("RGB", (512,512), color=0)
    hm = HeatMap(hm_cord)
    hit_img = hm.heatmap(base=hm_bg, r = 20) 
    hit_img = cv2.cvtColor(np.asarray(hit_img),cv2.COLOR_RGB2BGR)
    #hit_img = cv2.applyColorMap(np.asarray(hit_img), cv2.COLORMAP_JET)
    hit_img = hit_img.transpose((1,0,2)) #change X and Y
    hit_img = cv2.addWeighted(hit_img, 0.5, fm_img, 0.5, 0) 
    plt.imshow(hit_img)
    plt.axis('off')
    plt.savefig('/data/pycode/FundusDR/imgs/IDRiD_46_lesion.jpg', dpi = 400)


def VisMap():
    print('********************load model********************')
    if args.model == 'SANet':
        model = SANet(num_classes=N_CLASSES).cuda()
        CKPT_PATH = config['CKPT_PATH'] + args.model + '_best_model_v1.0.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model.load_state_dict(checkpoint) #strict=False
            print("=> Loaded well-trained checkpoint from: "+CKPT_PATH)
    else: 
        print('No required model')
        return #over
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    model.eval()#turn to test mode
    print('******************** load model succeed!********************')

    query_img_path = '/data/pycode/FundusDR/imgs/IDRiD_53.jpg' #IDRiD_53.jpg, IDRiD_46.jpg, IDRiD_09.jpg
    transform_seq = transforms.Compose([
        transforms.Resize((config['TRAN_SIZE'], config['TRAN_SIZE'])),
        transforms.ToTensor() #to tesnor [0,1]
    ])
    query_img = Image.open(query_img_path).convert('RGB')

    #infer feature map
    
    var_image = torch.autograd.Variable(torch.unsqueeze(transform_seq(query_img),0)).cuda()
    x, out, map_bb, map_sa = model(var_image)

    heat_map = map_sa.data.cpu().squeeze(0).sum(dim=0) #1024*8*8
    heat_map = heat_map.squeeze(0).numpy()#.permute(1, 2, 0)
    heat_map = heat_map - heat_map.min()
    heat_map = heat_map / heat_map.max()

    #plot heat map
    """
    sns_temp = sns.heatmap(heat_map, cmap="hot_r") #hot_r #OrRd
    sns_temp = sns_temp.get_figure()
    sns_temp.savefig('/data/pycode/FundusDR/imgs/IDRiD_53_sns_22.jpg', dpi = 400)
    """
   
    #plot overlay image
    """
    query_img =  np.asarray(query_img)
    width, height = query_img.shape[0],query_img.shape[1]
    heat_map = cv2.resize(heat_map,(height, width))
    heat_map = np.uint8(heat_map * 255.0)
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET) #L to RGB
    output_img = cv2.addWeighted(query_img, 0.7, heat_map, 0.3, 0)
    plt.imshow(output_img)
    plt.axis('off')
    plt.savefig('/data/pycode/FundusDR/imgs/IDRiD_53_map_sa_22.jpg', dpi = 400)

    """

    """
    query_img =  np.asarray(query_img)
    width, height = query_img.shape[0],query_img.shape[1]
    heat_map = cv2.resize(heat_map,(height, width))

    mask_ma = os.path.join('/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TrainingSet/Microaneurysms/IDRiD_53_MA.tif')
    mask_ma = Image.open(mask_ma).convert('L') #0,255
    mask_he = os.path.join('/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TrainingSet/Haemorrhages/IDRiD_53_HE.tif')
    mask_he = Image.open(mask_he).convert('L')

    heat_map = heat_map*(np.array(mask_he) + np.array(mask_ma))
    heat_map = np.uint8(heat_map)
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET) #L to RGB
    output_img = cv2.addWeighted(query_img, 0.7, heat_map, 0.3, 0)
    plt.imshow(output_img)
    plt.axis('off')
    plt.savefig('/data/pycode/FundusDR/imgs/IDRiD_53_map_sa_22.jpg', dpi = 400)
    """
    query_img = np.asarray(query_img)
    width, height = query_img.shape[0],query_img.shape[1]
    heat_map = cv2.resize(heat_map,(height, width))
    #heat_map = np.uint8(heat_map * 255.0)

    mask_ma = os.path.join('/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TrainingSet/Microaneurysms/IDRiD_53_MA.tif')
    mask_ma = Image.open(mask_ma).convert('L') #0,255
    #idx_ma = np.array(mask_ma).nonzero()
    mask_he = os.path.join('/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TrainingSet/Haemorrhages/IDRiD_53_HE.tif')
    mask_he = Image.open(mask_he).convert('L')
    #idx_hz = np.array(mask_he).nonzero()

    heat_map = heat_map*(np.array(mask_he) + np.array(mask_ma))
    idx_hm = np.argwhere(heat_map>0) #return nonzero index
    #idx_hm = np.where(heat_map>0)
    #hm_cord= ([idx_hm[0],idx_hm[1],heat_map[idx_hm[0],idx_hm[1]]])
    hm_cord = []
    for xy in idx_hm:
        hm_cord.append([xy[0], xy[1], heat_map[xy[0], xy[1]]])

    #https://www.cnblogs.com/lzhu/p/11738920.html
    hm_bg = Image.new("RGB", (query_img.shape[1], query_img.shape[0]), color=0)
    hm = HeatMap(hm_cord)
    hit_img = hm.heatmap(base=hm_bg, r = 10) 
    hit_img = cv2.cvtColor(np.asarray(hit_img),cv2.COLOR_RGB2BGR)
    image3 = cv2.addWeighted(hit_img, 0.3, query_img, 0.7, 0) 
    cv2.imwrite('/data/pycode/FundusDR/imgs/IDRiD_53_map_sa_22.jpg',image3)
    
    
def main():
    #vis_seg_map()
    vis_cls_map()

if __name__ == '__main__':
    main()
