# encoding: utf-8
"""
Attention-Guided Network for ChesstXRay 
Author: Jason.Fang
Update time: 16/01/2021
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

#define myself
#from config import *
#construct model
class ImageClassifier(nn.Module):
    def __init__(self, num_classes, is_pre_trained=True):
        super(ImageClassifier, self).__init__()
        self.dense_net_121 = torchvision.models.densenet121(pretrained=is_pre_trained)
        num_fc_kernels = self.dense_net_121.classifier.in_features #1024
        self.dense_net_121.classifier = nn.Sequential(nn.Linear(num_fc_kernels, num_classes), nn.Sigmoid())
        self.msa = MultiScaleAttention()
        self.gem = GeneralizedMeanPooling()

    def forward(self, x):
        #x: N*C*W*H
        """
        x = self.msa(x) * x
        x = self.dense_net_121(x)
        return x
        """
        x = self.msa(x) * x
        x = self.dense_net_121.features(x)
        #x = self.gem(x).view(x.size(0), -1)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7, stride=1).view(x.size(0), -1)
        out = self.dense_net_121.classifier(x)

        return x, out
        
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

#https://github.com/naver/deep-image-retrieval/blob/master/dirtorch/nets/layers/pooling.py
#https://arxiv.org/pdf/1711.02512.pdf
class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        #self.p = float(norm)
        self.p = nn.Parameter(torch.ones(1) * norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.p) + ', ' \
            + 'output_size=' + str(self.output_size) + ')'


#https://openaccess.thecvf.com/content_ICCV_2017/papers/Noh_Large-Scale_Image_Retrieval_ICCV_2017_paper.pdf
#https://github.com/prismformore/delf-pytorch
class RegionClassifier(nn.Module):
    def __init__(self, num_classes, is_pre_trained=True):
        super(RegionClassifier, self).__init__()
        self.msa = MultiScaleAttention()
        self.resbackbone = ResBackbone(is_pre_trained=is_pre_trained)
        self.fc = nn.Sequential(nn.Linear(1024, num_classes), nn.Sigmoid())
        self.attention_layers = nn.Sequential(BasicBlock(1024, 512, 1), nn.Conv2d(512, 1, 1))
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        #x: N*C*W*H
        x = self.msa(x) * x
        x = self.resbackbone(x)
        attention_score = self.attention_layers(x)
        attention_prob = self.softplus(attention_score)
        attention_feature_map = F.normalize(x, p=2, dim=1)  # l2 normalize per channel
        #attention_feature_map = x
        attention_feat = torch.mean(torch.mean(attention_prob * attention_feature_map, dim=2, keepdim=True), dim=3, keepdim=True) #max
        attention_feat = attention_feat.view(attention_feat.shape[0], -1)
        out = self.fc(attention_feat)

        return attention_feat, out

class ResBackbone(nn.Module):
    def __init__(self, is_pre_trained=True):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=is_pre_trained)
        self.pre_conv = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        self.features = nn.Sequential(
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.features(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

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
    img = torch.rand(32, 3, 224, 224)#.to(torch.device('cuda:%d'%4))
    var_img = torch.autograd.Variable(img).to(torch.device('cuda:%d'%4))

    model_img = ImageClassifier(num_classes=14, is_pre_trained=True).to(torch.device('cuda:%d'%4))
    fc_fea_img, out_img = model_img(var_img)

    model_roi = RegionClassifier(num_classes=14, is_pre_trained=True).to(torch.device('cuda:%d'%4))
    fc_fea_roi, out_roi = model_roi(var_img)

    model_fusion = FusionClassifier(input_size = 2048, output_size = 14).to(torch.device('cuda:%d'%4))
    fc_fea_fusion = torch.cat((fc_fea_img,fc_fea_roi), 1)
    var_fusion = torch.autograd.Variable(fc_fea_fusion).to(torch.device('cuda:%d'%4))
    out_fusion = model_fusion(var_fusion)
    print(out_fusion.size())

