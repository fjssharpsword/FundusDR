# encoding: utf-8
"""
Attention-Guided Network for ChesstXRay 
Author: Jason.Fang
Update time: 11/16/2020
"""
import re
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from skimage.measure import label
import cv2
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

#construct model
class ATNet(nn.Module):
    def __init__(self, num_classes, is_pre_trained):
        super(ATNet, self).__init__()
        self.dense_net_121 = torchvision.models.densenet121(pretrained=is_pre_trained)
        num_fc_kernels = self.dense_net_121.classifier.in_features #1024
        self.dense_net_121.classifier = nn.Sequential(nn.Linear(num_fc_kernels, num_classes), nn.Sigmoid())
        self.msa = MultiScaleAttention()
        self.fc = nn.Conv2d(num_fc_kernels, num_classes, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        #x: 3*256*256
        
        x = self.msa(x) * x
        out = self.dense_net_121(x) 
        return out
        """
        x = self.msa(x) * x
        x = self.dense_net_121.features(x) #output: 1024*8*8
        x = self.fc(x) #1024*8*8->14*8*8 
        x = self.sigmoid(x)
        x = x.view(x.size(0),x.size(1),x.size(2)*x.size(3)) #14*64
        tr_out = torch.mean(x, dim=1, keepdim=True).squeeze()
        bce_out = torch.mean(x, dim=2, keepdim=True).squeeze()
        return tr_out, bce_out
        """
         
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

"""
class MultiScaleAttention(nn.Module):#spatial attention module
    def __init__(self):
        super(MultiScaleAttention, self).__init__()
        self.aggConv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.sigmoid(self.aggConv(x))

        return x
"""    

if __name__ == "__main__":
    #for debug   
    x = torch.rand(10, 3, 256, 256).to(torch.device('cuda:%d'%7))
    model = ATNet(num_classes=5, is_pre_trained=True).to(torch.device('cuda:%d'%7))
    tr_out, bce_out = model(x)
    print(tr_out.size())
    print(bce_out.size())