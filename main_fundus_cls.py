# encoding: utf-8
"""
Training implementation for Fundus DR dataset  
Author: Jason.Fang
Update time: 18/11/2021
"""
import re
import sys
import os
import cv2
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import math
from thop import profile
from tensorboardX import SummaryWriter
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, confusion_matrix
#define by myself
from datasets.FundusDR_cls import get_train_dataloader, get_validation_dataloader, get_test_dataloader
from nets.densenet import densenet121, densenet169, TorchDenseNet121
from nets.resnet import resnet50
#config
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
max_epoches = 30
BATCH_SIZE = 16*8
CLASS_NAMES = ['No', 'Mild', 'Moderate', 'Severe', 'Proliferative']
CKPT_PATH = '/data/pycode/FundusDR/ckpt/resnet_cls.pkl' 
#nohup python main_fundus_cls.py > log/resnet_cls.log 2>&1 &
def Train():
    print('********************load data********************')
    train_loader = get_train_dataloader(batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader = get_validation_dataloader(batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    print ('==>>> SIIM trainning batch number: {}'.format(len(train_loader)))
    print ('==>>> total validation batch number: {}'.format(len(val_loader)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    model = resnet50(pretrained=False, num_classes=len(CLASS_NAMES)) 
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: " + CKPT_PATH)
    model = nn.DataParallel(model).cuda()  # make model available multi GPU cores training    
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    optimizer_model = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    lr_scheduler_model = lr_scheduler.StepLR(optimizer_model , step_size = 10, gamma = 1)
    criterion = nn.BCELoss().cuda() #nn.CrossEntropyLoss().cuda()
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    #log_writer = SummaryWriter('/data/tmpexec/tensorboard-log') #--port 10002, start tensorboard
    AUROC_best = 0.50
    for epoch in range(max_epoches):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , max_epoches))
        print('-' * 10)
        model.train()  #set model to training mode
        loss_train = []
        with torch.autograd.enable_grad():
            for batch_idx, (img, lbl) in enumerate(train_loader):
                #forward
                var_image = torch.autograd.Variable(img).cuda()
                var_label = torch.autograd.Variable(lbl).cuda()
                var_out = model(var_image)
                # backward and update parameters
                optimizer_model.zero_grad()
                loss_tensor = criterion.forward(var_out, var_label) 
                loss_tensor.backward()
                optimizer_model.step()
                #show 
                loss_train.append(loss_tensor.item())
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(loss_train) ))

        model.eval()#turn to test mode
        gt = torch.FloatTensor()
        pred = torch.FloatTensor()
        with torch.autograd.no_grad():
            for batch_idx, (image, label) in enumerate(val_loader):
                var_image = torch.autograd.Variable(image).cuda()
                var_output = model(var_image)#forward
                gt = torch.cat((gt, label), 0)
                pred = torch.cat((pred, var_output.data.cpu()), 0)
                sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
                sys.stdout.flush()
        gt_np = gt.numpy()
        pred_np = pred.numpy() 
        AUROCs = []
        for i in range(len(CLASS_NAMES)):
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        print("\r Epoch: %5d Validation average AUROC = %.2f" % ( epoch + 1, np.mean(AUROCs)*100 ) )

        if AUROC_best < np.mean(AUROCs):
            AUROC_best = np.mean(AUROCs)
            torch.save(model.module.state_dict(), CKPT_PATH) #Saving torch.nn.DataParallel Models
            print(' Epoch: {} model has been already save!'.format(epoch + 1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

def Test():
    print('********************load data********************')
    test_loader = get_test_dataloader(batch_size=32, shuffle=False, num_workers=8)
    print ('==>>> total test batch number: {}'.format(len(test_loader)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    model = resnet50(pretrained=False, num_classes=len(CLASS_NAMES)).cuda() 
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: " + CKPT_PATH)
    model.eval()#turn to test mode
    print('********************load model succeed!********************')

    print('******* begin testing!*********')
    gt = torch.FloatTensor()
    pred = torch.FloatTensor()
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(test_loader):
            var_image = torch.autograd.Variable(image).cuda()
            var_output = model(var_image)#forward
            gt = torch.cat((gt, label), 0)
            pred = torch.cat((pred, var_output.data.cpu()), 0)
            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()
    gt_np = gt.numpy()
    pred_np = pred.numpy() 
    AUROCs = []
    for i in range(len(CLASS_NAMES)):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i])) 
    print("\rTestset average AUROC = %.2f" % ( np.mean(AUROCs)*100 ) )
    for i in range(len(CLASS_NAMES)):
        print("\r {} : AUROC = {:.2f}".format( CLASS_NAMES[i], AUROCs[i]*100 ) )

def main():
    Train()
    Test()

if __name__ == '__main__':
    main()