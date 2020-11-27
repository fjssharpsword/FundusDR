import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import time
import random
from torch import Tensor
from typing import Tuple


class TripletRankingLoss(nn.Module):
    
    def __init__(self, m=0.1):
        super(TripletRankingLoss, self).__init__()
        self.m = m 
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    #sampling one triplet samples for each sample
    def _selTripletSamples(self, pred:Tensor, gt:Tensor)-> Tuple[Tensor, Tensor, Tensor]:
        anchor = torch.FloatTensor().cuda()
        positive = torch.FloatTensor().cuda()
        negative = torch.FloatTensor().cuda()

        #for id_a, (prob, label) in enumerate(zip(pred, gt)):
        for id_a, (label) in enumerate(gt):
            col = torch.where(label==1)[0]
            rows_pos = torch.where(gt[:, col]==1)[0]
            rows_neg = torch.where(gt[:, col]==0)[0]
            if len(rows_pos)>0 and len(rows_neg>0):
                #anchor
                anchor = torch.cat((anchor, pred[id_a,:].unsqueeze(0).cuda()), 0) 
                #positive
                id_p = random.sample(list(rows_pos), 1)[0]
                positive = torch.cat((positive, pred[id_p,:].unsqueeze(0).cuda()), 0)
                #negative
                id_n = random.sample(list(rows_neg), 1)[0]
                negative = torch.cat((negative, pred[id_n,:].unsqueeze(0).cuda()), 0)

        return anchor, positive, negative

    def forward(self, pred, gt):
        #gt: batchsize*num_classes
        #fea: batchsize*length of vector
        anchor, positive, negative = self._selTripletSamples(pred, gt)
        if (len(anchor)>0):
            cos_v = self.cos(anchor, positive) - self.cos(anchor, negative) + self.m
            loss = torch.where(cos_v<0, torch.zeros_like(cos_v), cos_v)#max(cos_v, 0)
            loss = torch.mean(loss).requires_grad_()
        else:
            loss = torch.float(0.0)
        return loss

    """
    #sampling all pospair and negpair
    def __init__(self, scale=1, margin=0.25, similarity='cos', **kwargs):
        super(TripletRankingLoss, self).__init__()
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
    """
    
if __name__ == "__main__":
    #for debug   
    gt = torch.zeros(512, 5)
    pred = torch.rand(512, 64)
    for i in range(512):#generate 1 randomly
        col = random.randint(0,4)
        gt[i, col] = 1
    #a = torch.rand(512, 14)
    #p = torch.rand(512, 14)
    #n = torch.rand(512, 14)
    trl = TripletRankingLoss()
    loss = trl(pred, gt)
    print(loss)
