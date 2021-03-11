# encoding: utf-8
"""
Mirror-attention for combating ambiguity in fundus image retrieval.
Author: Jason.Fang
Update time: 10/03/2021
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
#define by myself

#construct model
class MANet(nn.Module):
    def __init__(self, is_pre_trained):
        super(MANet, self).__init__()
        #backbone
        self.dense_net_121 = torchvision.models.densenet121(pretrained=is_pre_trained)
        self.in_ch = self.dense_net_121.classifier.in_features #1024
        #for negative
        self.f_n = nn.Sequential(
            nn.Conv2d(self.in_ch, self.in_ch, (1, 1), (1, 1)),
            nn.BatchNorm2d(self.in_ch),
            nn.ReLU())
        self.f_an = nn.Sequential(
            nn.Conv2d(self.in_ch, self.in_ch, (1, 1), (1, 1)),
            nn.BatchNorm2d(self.in_ch),
            nn.ReLU())
        #for postive
        self.f_p = nn.Sequential(
            nn.Conv2d(self.in_ch, self.in_ch, (1, 1), (1, 1)),
            nn.BatchNorm2d(self.in_ch),
            nn.ReLU())
        self.f_ap = nn.Sequential(
            nn.Conv2d(self.in_ch, self.in_ch, (1, 1), (1, 1)),
            nn.BatchNorm2d(self.in_ch),
            nn.ReLU())
        
    def forward(self, x_a, x_p, x_n):
        #x: 3*256*256
        x_a = self.dense_net_121.features(x_a) #output: 1024*8*8
        x_p = self.dense_net_121.features(x_p)
        x_n = self.dense_net_121.features(x_n)

        B, C, H, W = x_a.shape
        x_ap = self.f_ap(x_a).view(B, self.in_ch, H * W)  # B * in_ch * N, where N = H*W
        x_an = self.f_an(x_a).view(B, self.in_ch, H * W)
        x_n = self.f_n(x_n).view(B, self.in_ch, H * W)
        x_p = self.f_p(x_p).view(B, self.in_ch, H * W)

        z = torch.bmm(f_x.permute(0, 2, 1), g_x)  # B * N * N, where N = H*W



        return x
         

if __name__ == "__main__":
    #for debug   
    x = torch.rand(10, 3, 256, 256)
    model = MANet(num_classes=5, is_pre_trained=True)
    out = model(x)
    print(out.size())