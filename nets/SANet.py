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
class SANet(nn.Module):
    def __init__(self, num_classes):
        super(SANet, self).__init__()
        #backbone
        self.spa = SpatialAttention()
        self.dense_net_121 = torchvision.models.densenet121(pretrained=True)
        in_ch = self.dense_net_121.classifier.in_features #1024
        self.sfa = SelfAttention(in_ch=in_ch, k=2)
        self.gem = GeM()
        self.classifier = nn.Sequential(nn.Linear(in_ch, num_classes), nn.Sigmoid())
        
    def forward(self, x):
        #Backbone: extract convolutional feature maps, 1024*8*8
        x = self.spa(x)*x 
        x = self.dense_net_121.features(x) 

        map_bb = x

        x = self.sfa(x)

        map_sa = x

        x = self.gem(x).view(x.size(0), -1)
        out = self.classifier(x)

        return x, out, map_bb, map_sa

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

class SelfAttention(nn.Module): #self-attention block
    def __init__(self, in_ch, k):
        super(SelfAttention, self).__init__()

        self.in_ch = in_ch
        self.out_ch = in_ch
        self.mid_ch = in_ch // k

        print('Num channels:  in    out    mid')
        print('               {:>4d}  {:>4d}  {:>4d}'.format(self.in_ch, self.out_ch, self.mid_ch))

        self.f = nn.Sequential(
            nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1)),
            nn.BatchNorm2d(self.mid_ch),
            nn.ReLU())
        self.g = nn.Sequential(
            nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1)),
            nn.BatchNorm2d(self.mid_ch),
            nn.ReLU())
        self.h = nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1))
        self.v = nn.Conv2d(self.mid_ch, self.out_ch, (1, 1), (1, 1))

        self.softmax = nn.Softmax(dim=-1)

        for conv in [self.f, self.g, self.h]: 
            conv.apply(weights_init)
        self.v.apply(constant_init)

    def forward(self, x):
        B, C, H, W = x.shape

        f_x = self.f(x).view(B, self.mid_ch, H * W)  # B * mid_ch * N, where N = H*W
        g_x = self.g(x).view(B, self.mid_ch, H * W)  # B * mid_ch * N, where N = H*W
        h_x = self.h(x).view(B, self.mid_ch, H * W)  # B * mid_ch * N, where N = H*W

        z = torch.bmm(f_x.permute(0, 2, 1), g_x)  # B * N * N, where N = H*W
        attn = self.softmax((self.mid_ch ** -.50) * z)

        z = torch.bmm(attn, h_x.permute(0, 2, 1))  # B * N * mid_ch, where N = H*W
        z = z.permute(0, 2, 1).view(B, self.mid_ch, H, W)  # B * mid_ch * H * W

        z = self.v(z)
        z = z + x

        return z

## Kaiming weight initialisation
def weights_init(module):
    if isinstance(module, nn.ReLU):
        pass
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        pass
def constant_init(module):
    if isinstance(module, nn.ReLU):
        pass
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.constant_(module.weight.data, 0.0)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        pass

class SpatialAttention(nn.Module):#spatial attention layer
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

if __name__ == "__main__":
    #for debug  
    x = torch.rand(10, 3, 256, 256)
    model = SANet(num_classes=5)
    x, out = model(x)
    print(x.size())
    print(out.size())
