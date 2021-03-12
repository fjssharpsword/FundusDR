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
from torch.nn.parameter import Parameter
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from skimage.measure import label
import cv2
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
#define by myself

#construct model
class MANet(nn.Module):
    def __init__(self, is_pre_trained):
        super(MANet, self).__init__()
        #backbone
        self.dense_net_121 = torchvision.models.densenet121(pretrained=True)
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
        #function
        self.is_pre_trained = is_pre_trained
        self.softmax = nn.Softmax(dim=-1)
        self.gem = GeM()
        
    def forward(self, x_a, x_p=None, x_n=None):
        if self.is_pre_trained:#True, for train
            #Backbone: extract convolutional feature maps, 1024*8*8
            x_a = self.dense_net_121.features(x_a) 
            x_p = self.dense_net_121.features(x_p)
            x_n = self.dense_net_121.features(x_n)
            #Mirror-attention
            B, C, H, W = x_a.shape
            x_an = self.f_an(x_a).view(B, self.in_ch, H * W) # B * C * HW
            x_ap = self.f_ap(x_a).view(B, self.in_ch, H * W)  
            x_nn = self.f_n(x_n).view(B, self.in_ch, H * W)
            x_pp = self.f_p(x_p).view(B, self.in_ch, H * W)

            z_n = torch.bmm(x_nn.permute(0, 2, 1), x_an)  # B * HW * HW
            z_n = self.softmax(z_n)
            z_p = torch.bmm(x_pp.permute(0, 2, 1), x_ap)
            z_p = self.softmax(z_p)

            x_a = x_a.view(B, self.in_ch, H * W)# B * C * HW
            x_an = torch.bmm(x_a, z_n).view(B, self.in_ch, H, W)# B * C * H * W
            x_ap = torch.bmm(x_a, z_p).view(B, self.in_ch, H, W)
            x_p = x_p.view(B, self.in_ch, H * W)# B * C * HW
            x_p = torch.bmm(x_p, z_p).view(B, self.in_ch, H, W)
            x_n = x_n.view(B, self.in_ch, H * W)# B * C * HW
            x_n = torch.bmm(x_n, z_n).view(B, self.in_ch, H, W)

            #GeMLayer:
            x_an = self.gem(x_an).view(x_an.size(0), -1)
            x_n = self.gem(x_n).view(x_n.size(0), -1)
            x_ap = self.gem(x_ap).view(x_ap.size(0), -1)
            x_p = self.gem(x_p).view(x_p.size(0), -1)

            return x_an, x_n, x_ap, x_p
        else:#False, for test
            x_a = self.dense_net_121.features(x_a) 
            x_p = self.dense_net_121.features(x_a) 
            B, C, H, W = x_a.shape
            x_ap = self.f_ap(x_a).view(B, self.in_ch, H * W)
            x_pp = self.f_p(x_p).view(B, self.in_ch, H * W)
            z_p = torch.bmm(x_pp.permute(0, 2, 1), x_ap)
            z_p = self.softmax(z_p)
            x_ap = torch.bmm(x_a, z_p).view(B, self.in_ch, H, W)
            x_ap = self.gem(x_ap).view(x_ap.size(0), -1)
            return x_ap

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)
    # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    """
    def __init__(self, p=2, margin=10):
        super(ContrastiveLoss, self).__init__()
        self.dist = nn.PairwiseDistance(p=2)
        self.margin = margin
        #self.rankloss = nn.MarginRankingLoss(margin=margin)

    def forward(self, x_an, x_n, x_ap, x_p):
        dis_an = self.dist(x_an, x_n)
        dis_ap = self.dist(x_ap, x_p)
        #target = torch.ones(len(dis_an))#dis_an>dis_ap, lbl=1
        #dis_loss = self.rankloss(dis_an, dis_ap, target) 
        dis_loss =torch.mean(torch.clamp((self.margin - (dis_an-dis_ap)), min=0))
        return dis_loss

if __name__ == "__main__":
    #for debug  
    a = torch.rand(10, 3, 256, 256)
    p = torch.rand(10, 3, 256, 256)
    n = torch.rand(10, 3, 256, 256)
    model = MANet(is_pre_trained=True)
    x_an, x_n, x_ap, x_p = model(a, p, n)
    conloss = ContrastiveLoss()
    loss = conloss(x_an, x_n, x_ap, x_p)
    print(loss)
    #test
    model = MANet(is_pre_trained=False)
    x_a = model(a)
    print(x_a.size())
