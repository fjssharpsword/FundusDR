from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import re
import sys
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from config import *

def compute_ROCCurve(gt, pred):
    #fpr = 1-Specificity, tpr=Sensitivity
    np.set_printoptions(suppress=True) #to float
    thresholds = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    color_name =['r','b','k','y','c']
    for i in range(N_CLASSES):
        fpr, tpr, threshold = roc_curve(gt_np[:, i], pred_np[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, c = color_name[i], ls = '--', label = u'{}-AUROC{:.4f}'.format(CLASS_NAMES[i],auc_score))
        #select the prediction threshold
        idx = np.where(tpr>auc_score)[0][0]
        thresholds.append(threshold[idx])

    #plot and save
    plt.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right')
    plt.title('Fundus DR')
    plt.savefig(config['img_path']+'ROCCurve.jpg')

    return thresholds

def compute_AUCs(gt, pred):
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

def compute_IoUs(xywh1, xywh2):
    x1, y1, w1, h1 = xywh1
    x2, y2, w2, h2 = xywh2

    dx = min(x1+w1, x2+w2) - max(x1, x2)
    dy = min(y1+h1, y2+h2) - max(y1, y2)
    intersection = dx * dy if (dx >=0 and dy >= 0) else 0.
    
    union = w1 * h1 + w2 * h2 - intersection
    IoUs = intersection / union
    
    return IoUs