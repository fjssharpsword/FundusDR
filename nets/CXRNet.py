# encoding: utf-8
"""
Attention-Guided Network for ChesstXRay 
Author: Jason.Fang
Update time: 10/12/2020
"""
import sys
import re
import numpy as np
import random
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from skimage.measure import label as skmlabel
import cv2
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

#construct model
class CXRClassifier(nn.Module):
    def __init__(self, num_classes, is_pre_trained=True, is_roi=False):
        super(CXRClassifier, self).__init__()
        self.dense_net_121 = torchvision.models.densenet121(pretrained=is_pre_trained)
        num_fc_kernels = self.dense_net_121.classifier.in_features #1024
        self.dense_net_121.classifier = nn.Sequential(nn.Linear(num_fc_kernels, num_classes), nn.Sigmoid())
        self.msa = MultiScaleAttention()
        #self.sa = SpatialAttention()
        self.is_roi =  is_roi
        
    def forward(self, x):
        #x: N*C*W*H
        """
        x = self.msa(x) * x
        x = self.dense_net_121(x)
        return x
        """
        if self.is_roi == False: #for image training
            x = self.msa(x) * x
        conv_fea = self.dense_net_121.features(x)
        out = F.relu(conv_fea, inplace=True)
        fc_fea = F.avg_pool2d(out, kernel_size=7, stride=1).view(conv_fea.size(0), -1)
        out = self.dense_net_121.classifier(fc_fea)
        return conv_fea, fc_fea, out
        
        
class MultiScaleAttention(nn.Module):#multi-scal attention module
    def __init__(self):
        super(MultiScaleAttention, self).__init__()
        
        self.scaleConv1 = nn.Conv2d(3, 3, kernel_size=5, padding=2, bias=False)
        self.scaleConv2 = nn.Conv2d(3, 3, kernel_size=9, padding=4, bias=False)
        
        self.aggConv = nn.Conv2d(6, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid() 
        
    def forward(self, x):
        out_max, _ = torch.max(x, dim=1, keepdim=True)
        out_avg = torch.mean(x, dim=1, keepdim=True)
        
        out1 = self.scaleConv1(x)
        out_max1, _ = torch.max(out1, dim=1, keepdim=True)
        out_avg1 = torch.mean(out1, dim=1, keepdim=True)
        
        out2 = self.scaleConv2(x)
        out_max2, _ = torch.max(out2, dim=1, keepdim=True)
        out_avg2 = torch.mean(out2, dim=1, keepdim=True)

        x = torch.cat([out_max, out_avg, out_max1, out_avg1, out_max2, out_avg2], dim=1)
        x = self.sigmoid(self.aggConv(x))

        return x

class SpatialAttention(nn.Module):#spatial attention module
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.aggConv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.sigmoid(self.aggConv(x))
        return x  

class ROIGenerator(object):
    def __init__(self, TRANS_CROP=224):
        super(ROIGenerator, self).__init__()
        self.TRANS_CROP = TRANS_CROP
        self.transform_seq = transforms.Compose([
                                #transforms.Resize((256,256)),
                                #transforms.RandomCrop(224),
                                #transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                ])

    def ROIGeneration(self, ori_image, fm_cuda, weight_softmax, label):
        # feature map -> feature mask (using feature map to crop on the original image) -> crop -> patchs
        feature_conv = fm_cuda.data.cpu().numpy()
        size_upsample = (self.TRANS_CROP, self.TRANS_CROP) 
        bz, nc, h, w = feature_conv.shape

        patchs = torch.FloatTensor()

        for i in range(0, bz):
            feature = feature_conv[i]
            cam = feature.reshape((nc, h*w))
            class_idx = np.where(label[i]==1)[0]
            if len(class_idx)>0: 
                cam = weight_softmax[class_idx].dot(cam) #class activated map
            cam = cam.sum(axis=0)
            cam = cam.reshape(h,w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)

            heatmap_bin = self.binImage(cv2.resize(cam_img, size_upsample))
            heatmap_maxconn = self.selectMaxConnect(heatmap_bin)
            heatmap_mask = heatmap_bin * heatmap_maxconn

            ind = np.argwhere(heatmap_mask != 0)
            minh = min(ind[:,0])
            minw = min(ind[:,1])
            maxh = max(ind[:,0])
            maxw = max(ind[:,1])
            
            # to ori image 
            image = ori_image[i].numpy().reshape(224,224,3)
            image = image[int(224*0.334):int(224*0.667),int(224*0.334):int(224*0.667),:]

            image = cv2.resize(image, size_upsample)
            image_crop = image[minh:maxh,minw:maxw,:] * 255 # because image was normalized before
            image_crop = cv2.resize(image_crop, size_upsample)
            image_crop = self.transform_seq(Image.fromarray(image_crop.astype('uint8')).convert('RGB')) 
            
            #img_variable = torch.autograd.Variable(image_crop)
            patchs = torch.cat((patchs,image_crop.unsqueeze(0)),0)

        return patchs


    def binImage(self, heatmap):
        _, heatmap_bin = cv2.threshold(heatmap , 0 , 255 , cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # t in the paper
        #_, heatmap_bin = cv2.threshold(heatmap , 178 , 255 , cv2.THRESH_BINARY)
        return heatmap_bin

    def selectMaxConnect(self, heatmap):
        labeled_img, num = skmlabel(heatmap, connectivity=2, background=0, return_num=True)    
        max_label = 0
        max_num = 0
        for i in range(1, num+1):
            if np.sum(labeled_img == i) > max_num:
                max_num = np.sum(labeled_img == i)
                max_label = i
        lcc = (labeled_img == max_label)
        if max_num == 0:
            lcc = (labeled_img == -1)
        lcc = lcc + 0
        return lcc 

class FusionClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(FusionClassifier, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, fusion_var):
        out = self.fc(fusion_var)
        out = self.Sigmoid(out)

        return out

if __name__ == "__main__":
    #for debug   
    img = torch.rand(32, 3, 224, 224).to(torch.device('cuda:%d'%4))
    label = torch.zeros(32, 14)
    for i in range(32):#generate 1 randomly
        ones_n = random.randint(1,2)
        col = [random.randint(0,13) for _ in range(ones_n)]
        label[i, col] = 1
    model_img = CXRClassifier(num_classes=14, is_pre_trained=True, is_roi=False).to(torch.device('cuda:%d'%4))
    conv_fea_img, fc_fea_img, out_img = model_img(img)
    roigen = ROIGenerator()
    cls_weights = list(model_img.parameters())
    weight_softmax = np.squeeze(cls_weights[-5].data.cpu().numpy())
    roi = roigen.ROIGeneration(img.cpu(), conv_fea_img, weight_softmax, label)
    model_roi = CXRClassifier(num_classes=14, is_pre_trained=True, is_roi=True).to(torch.device('cuda:%d'%4))
    var_roi = torch.autograd.Variable(roi).to(torch.device('cuda:%d'%4))
    _, fc_fea_roi, out_roi = model_roi(var_roi)
    model_fusion = FusionClassifier(input_size = 2048, output_size = 14).to(torch.device('cuda:%d'%4))
    fc_fea_fusion = torch.cat((fc_fea_img,fc_fea_roi), 1)
    var_fusion = torch.autograd.Variable(fc_fea_fusion).to(torch.device('cuda:%d'%4))
    out_fusion = model_fusion(var_fusion)
    print(out_fusion.size())

