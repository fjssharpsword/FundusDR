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
    def __init__(self):
        super(SANet, self).__init__()
        #backbone
        self.spa = SpatialAttention()
        self.dense_net_121 = torchvision.models.densenet121(pretrained=True)
        in_ch = self.dense_net_121.classifier.in_features #1024
        self.sfa = SelfAttention(in_ch=in_ch, k=2)
        self.gem = GeM()
        
    def forward(self, x):
        #Backbone: extract convolutional feature maps, 1024*8*8
        x = self.spa(x)*x 
        x = self.dense_net_121.features(x) 
        x = self.sfa(x)
        out = self.gem(x).view(x.size(0), -1)

        return out

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

class RelevanceDegreeLoss(nn.Module):
    """
    RelevanceDegreeLoss loss function.
    Input: 
          feats: (B, V), batchsize and image descriptor
          labels: (B,), number of lesions
    """
    def __init__(self, margin=1):
        super(RelevanceDegreeLoss, self).__init__()
        self.margin = margin #margin of each degree

    def forward(self, feats, labels):
        #similar matrix
        feats = F.normalize(feats)
        sim_mat = feats.mm(feats.t())
        #degree matrix
        mat_a = torch.matmul(labels, 1/torch.t(labels))
        mat_a = torch.where(mat_a<1, mat_a, torch.zeros_like(mat_a))
        mat_b = torch.matmul(1/labels, torch.t(labels))
        mat_b = torch.where(mat_b<1, mat_b, torch.zeros_like(mat_b))
        deg_mat = torch.add(mat_a, mat_b)
        deg_mat = deg_mat.triu(diagonal=1)
        #deg_mat = (deg_mat - 1).abs_()
        #sort
        sim_mat = sim_mat[deg_mat>0]
        deg_mat = deg_mat[deg_mat>0]
        _, indices = torch.sort(deg_mat, descending=True) #
        sim_mat = sim_mat[indices].unsqueeze(1)
        sim_mat = torch.matmul(sim_mat, 1/torch.t(sim_mat))
        sim_mat = sim_mat.triu(diagonal=1)
        sim_mat = sim_mat[sim_mat<1]
        rdloss = torch.log(torch.sum(sim_mat))
        
        return rdloss

if __name__ == "__main__":
    #for debug  
    #x = torch.rand(10, 3, 256, 256)
    #model = SANet()
    #out = model(x)
    #print(out.size())
    labels = torch.Tensor([9, 10, 7, 6, 8, 5, 4, 2, 3, 1]).unsqueeze(1).cuda()
    feats = torch.rand(10,1024).cuda()
    criterion = RelevanceDegreeLoss().cuda()
    rdloss  =  criterion(feats,labels)
    print(rdloss)
