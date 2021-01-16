# encoding: utf-8
"""
Combating Ambiguity for Diabetic Retinopathy Retrieval in Fundus Images by unifying Self-supervised and Supervised Learning.
Author: Jason.Fang
Update time: 15/01/2021
"""
import sys
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
from torchvision.ops import RoIAlign
#defined by myself

# define ConvAutoencoder architecture
class AENet(nn.Module):
    def __init__(self, num_classes=5):
        super(AENet, self).__init__()

        ## encoder layers ##
        # conv layer (depth from 3 --> 128), 3x3 kernels
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1)  
        # conv layer (depth from 128 --> 256), 3x3 kernels
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        # conv layer (depth from 256 --> 512), 3x3 kernels
        self.conv3 = nn.Conv2d(256, 512, 3, padding=1)
        # conv layer (depth from 512 --> 1024), 3x3 kernels
        self.conv4 = nn.Conv2d(512, 1024, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.AvgPool2d(2, 2)
        
        #latent layer
        self.dnpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.uppool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(128, 3, 2, stride=2)

        #common layer
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.msa = MultiScaleAttention()
        self.gem = GeneralizedMeanPooling()
        self.classifer =  nn.Sequential(nn.Linear(1024, num_classes), nn.Sigmoid())

    def forward(self, x):
        ##Attention Layer##
        x = self.msa(x)*x
        ## encode ##
        # add hidden layers with relu activation function
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x))) 

        #latent vector for retrieval and classification
        feat, indices  = self.dnpool(x)
        x = self.uppool(feat, indices)
        feat = self.gem(feat)
        feat = feat.view(feat.size(0),-1)
        out = self.classifer(feat)
        #vec = vec.view(vec.size(0), vec.size(1), vec.size(2)*vec.size(3))
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = self.relu(self.t_conv1(x))
        x = self.relu(self.t_conv2(x))
        x = self.relu(self.t_conv3(x))
        x = self.relu(self.t_conv4(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = self.sigmoid(x)

        return feat, out, x

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

#https://github.com/qianjinhao/circle-loss/blob/master/circle_loss.py
#sampling all pospair and negpair
class CircleLoss(nn.Module):
    def __init__(self, scale=1, margin=0.25, similarity='cos', **kwargs):
        super(CircleLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.similarity = similarity

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"

        mask = torch.matmul(labels, torch.t(labels))
        #mask = torch.where(mask==2, torch.zeros_like(mask), mask) #for multi-label
        pos_mask = mask.triu(diagonal=1)
        neg_mask = (mask - 1).abs_().triu(diagonal=1)
        if self.similarity == 'dot':
            sim_mat = torch.matmul(feats, torch.t(feats))
        elif self.similarity == 'cos':
            feats = F.normalize(feats)
            sim_mat = feats.mm(feats.t())
        else:
            raise ValueError('This similarity is not implemented.')

        pos_pair_ = sim_mat[pos_mask == 1]
        neg_pair_ = sim_mat[neg_mask == 1]
        #neg_pair_ = sim_mat[neg_mask == 1][0:len(pos_pair_)] #for sampling part normal 

        alpha_p = torch.relu(-pos_pair_ + 1 + self.margin)
        alpha_n = torch.relu(neg_pair_ + self.margin)
        margin_p = 1 - self.margin
        margin_n = self.margin
        loss_p = torch.sum(torch.exp(-self.scale * alpha_p * (pos_pair_ - margin_p)))
        loss_n = torch.sum(torch.exp(self.scale * alpha_n * (neg_pair_ - margin_n)))
        loss = torch.log(1 + loss_p * loss_n)
        return loss
        
#A CNN Variational Autoencoder in PyTorch
#https://github.com/sksq96/pytorch-vae/blob/master/vae.py 

if __name__ == "__main__":
    #for debug   
    x = torch.rand(2, 3, 224, 224)#.to(torch.device('cuda:%d'%7))
    model = AENet()#.to(torch.device('cuda:%d'%7))
    feat, out, x = model(x)
    print(feat.size())
    print(out.size())
    print(x.size())