# encoding: utf-8
"""
Mirror Attention-based Fine Triplet Loss for fundus image retrieval.
Author: Jason.Fang
Update time: 30/03/2021
"""
import re
import numpy as np
import pandas as pd
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
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(MANet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Sequential(OutConv(64, n_channels), nn.Sigmoid())

        #more blocks
        self.ma = MirrorAttention(in_ch=512, k=2)
        self.gem = GeM()
        self.classifier = nn.Sequential(nn.Linear(512*2, n_classes), nn.Sigmoid())

    def forward(self, xa, xh, xv):
        #down-encoder
        xa1 = self.inc(xa)
        xa2 = self.down1(xa1)
        xa3 = self.down2(xa2)
        xa4 = self.down3(xa3)
        xa5 = self.down4(xa4)

        xh1 = self.inc(xh)
        xh2 = self.down1(xh1)
        xh3 = self.down2(xh2)
        xh4 = self.down3(xh3)
        xh5 = self.down4(xh4)

        xv1 = self.inc(xv)
        xv2 = self.down1(xv1)
        xv3 = self.down2(xv2)
        xv4 = self.down3(xv3)
        xv5 = self.down4(xv4)

        #mirror attention
        y_ah, y_av =self.ma(xa5, xh5, xa5) 
        xh_feat = self.gem(y_ah).view(y_ah.size(0), -1)
        xv_feat = self.gem(y_av).view(y_av.size(0), -1)
        xa_feat = torch.cat((xh_feat, xv_feat), 1)#dim=1
        xa_clss = self.classifier(xa_feat)

        #up- decoder
        xh = self.up1(y_ah, xh4)
        xh = self.up2(xh, xh3)
        xh = self.up3(xh, xh2)
        xh = self.up4(xh, xh1)
        m_h = self.outc(xh)

        xv = self.up1(y_av, xv4)
        xv = self.up2(xv, xv3)
        xv = self.up3(xv, xv2)
        xv = self.up4(xv, xv1)
        m_v = self.outc(xv)
        
        return xa_feat, xa_clss, m_h, m_v

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

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

class FineTripletLoss(nn.Module):
    #sampling all pospair and negpair
    def __init__(self, scale=1, margin=0.25, similarity='cos', **kwargs):
        super(FineTripletLoss, self).__init__()
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
        """
        degs = feats.view(feats.size(0), 2, int(feats.size(1)/2))
        degs = F.cosine_similarity(degs[:,0,:], degs[:,1,:])
        degs = torch.unsqueeze(degs, 1)
        deg_mat = torch.mm(1/degs, torch.t(degs))
        deg_mat = deg_mat.mul(mask)
        pos_pair_deg = deg_mat[pos_mask == 1]
        """
        #loss_p = torch.sum(torch.exp(-pos_pair_deg * alpha_p * (pos_pair_ - margin_p)))

        alpha_p = torch.relu(-pos_pair_ + 1 + self.margin)
        alpha_n = torch.relu(neg_pair_ + self.margin)
        margin_p = 1 - self.margin
        margin_n = self.margin
        loss_p = torch.sum(torch.exp(-self.scale * alpha_p * (pos_pair_ - margin_p)))
        loss_n = torch.sum(torch.exp(self.scale * alpha_n * (neg_pair_ - margin_n)))
        loss = torch.log(1 + loss_p * loss_n)
        return loss

class MirrorAttention(nn.Module): #mirror-attention block
    def __init__(self, in_ch, k):
        super(MirrorAttention, self).__init__()

        self.in_ch = in_ch
        self.out_ch = in_ch
        self.mid_ch = in_ch // k

        print('Num channels:  in    out    mid')
        print('               {:>4d}  {:>4d}  {:>4d}'.format(self.in_ch, self.out_ch, self.mid_ch))

        self.f_a = nn.Sequential(
            nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1)),
            nn.BatchNorm2d(self.mid_ch),
            nn.ReLU())
        self.f_v = nn.Sequential(
            nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1)),
            nn.BatchNorm2d(self.mid_ch),
            nn.ReLU())
        self.f_h = nn.Sequential(
            nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1)),
            nn.BatchNorm2d(self.mid_ch),
            nn.ReLU())

        self.g_av = nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1))
        self.g_ah = nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1))

        self.f_av = nn.Conv2d(self.mid_ch, self.out_ch, (1, 1), (1, 1))
        self.f_ah = nn.Conv2d(self.mid_ch, self.out_ch, (1, 1), (1, 1))

        self.softmax = nn.Softmax(dim=-1)

        for conv in [self.f_a, self.f_v, self.f_h, self.g_av, self.g_ah]: 
            conv.apply(weights_init)
        for conv in [self.f_av, self.f_ah]: 
            conv.apply(constant_init)

    def forward(self, x_a, x_h, x_v):
        B, C, H, W = x_a.shape

        f_a = self.f_a(x_a).view(B, self.mid_ch, H * W)  # B * mid_ch * N, where N = H*W
        f_v = self.f_v(x_v).view(B, self.mid_ch, H * W)
        f_h = self.f_v(x_h).view(B, self.mid_ch, H * W)

        f_av = torch.bmm(f_v.permute(0, 2, 1), f_a)  # B * N * N, where N = H*W
        f_av = self.softmax((self.mid_ch ** -.50) * f_av)
        f_ah = torch.bmm(f_h.permute(0, 2, 1), f_a)
        f_ah = self.softmax((self.mid_ch ** -.50) * f_ah)

        g_av = self.g_av(x_a).view(B, self.mid_ch, H * W)
        g_ah = self.g_ah(x_a).view(B, self.mid_ch, H * W)

        f_v = torch.bmm(g_av, f_av).view(B, self.mid_ch, H, W)  # B * mid_ch * H * W
        y_av = self.f_av(f_v) + x_a
        f_h = torch.bmm(g_ah, f_ah).view(B, self.mid_ch, H, W) 
        y_ah = self.f_ah(f_h) + x_a

        return y_ah, y_av

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
    x_h = torch.rand(10, 3, 256, 256)
    x_v = torch.rand(10, 3, 256, 256)
    model = MANet(n_channels=3, n_classes=5)
    xa_feat, xa_clss, m_h, m_v = model(x,x_h, x_v)
    print(xa_feat.size())
    print(xa_clss.size())
    print(m_h.size())
    print(m_v.size())
