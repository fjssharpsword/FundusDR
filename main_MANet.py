# encoding: utf-8
"""
Training implementation for Mirror Attention
Author: Jason.Fang
Update time: 11/03/2021
"""
import re
import sys
import os
import cv2
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
from skimage.measure import label
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
#self-defined
from config import *
from utils.logger import get_logger
from datasets.KaggleDR import get_train_dataloader, get_validation_dataloader, get_test_dataloader
from nets.MANet import MANet, FineTripletLoss

#command parameters
parser = argparse.ArgumentParser(description='For FundusDR')
parser.add_argument('--model', type=str, default='MANet', help='MANet')
args = parser.parse_args()

#config
os.environ['CUDA_VISIBLE_DEVICES'] = config['CUDA_VISIBLE_DEVICES']
logger = get_logger(config['log_path'])

def Train():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=8)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'MANet':
        model = MANet(n_channels=3, n_classes=N_CLASSES).cuda()
        CKPT_PATH = config['CKPT_PATH'] + args.model + '_best_model.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model.load_state_dict(checkpoint) #strict=False
            print("=> Loaded well-trained checkpoint from: "+CKPT_PATH)
    else: 
        print('No required model')
        return #over
    model = nn.DataParallel(model).cuda()  # make model available multi GPU cores training    
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_model = lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)
    #define loss function
    bceloss = nn.BCELoss() 
    mseloss = nn.MSELoss()
    triloss = FineTripletLoss()
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    loss_min = float('inf')
    for epoch in range(config['MAX_EPOCHS']):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , config['MAX_EPOCHS']))
        print('-' * 10)
        model.train()  #set model to training mode
        loss_train = []
        with torch.autograd.enable_grad():
            for batch_idx, (img_a, img_h, img_v, label) in enumerate(dataloader_train):
                optimizer.zero_grad()
                #forward
                var_image_a = torch.autograd.Variable(img_a).cuda()
                var_image_h = torch.autograd.Variable(img_h).cuda()
                var_image_v = torch.autograd.Variable(img_v).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                var_feat, var_clss, mask_h, mask_v = model(var_image_a, var_image_h, var_image_v)
                # backward and update parameters
                loss_h = mseloss.forward(mask_h, var_image_h)
                loss_v = mseloss.forward(mask_v, var_image_v)
                loss_tri = triloss.forward(var_feat, var_label)
                #loss_cls = bceloss.forward(var_clss, var_label)

                loss_tensor = loss_h + loss_v + loss_tri
                loss_tensor.backward()
                optimizer.step()
                #show 
                loss_train.append(loss_tensor.item())
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(loss_train) ))

        # save checkpoint
        if loss_min > np.mean(loss_train):
            loss_min = np.mean(loss_train)
            torch.save(model.module.state_dict(), config['CKPT_PATH'] + args.model + '_best_model.pkl')
            print(' Epoch: {} model has been already save!'.format(epoch + 1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

def Test():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    dataloader_test = get_test_dataloader(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    print('********************load data succeed!********************')

    print('********************load model********************')
    if args.model == 'MANet':
        model = MANet(n_channels=3, n_classes=N_CLASSES).cuda()
        CKPT_PATH = config['CKPT_PATH'] + args.model + '_best_model.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model.load_state_dict(checkpoint) #strict=False
            print("=> Loaded well-trained checkpoint from: "+CKPT_PATH)
    else: 
        print('No required model')
        return #over
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    model.eval()#turn to test mode
    print('******************** load model succeed!********************')

    print('********************Build feature database!********************')
    tr_label = torch.FloatTensor().cuda()
    tr_feat = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (img_a, img_h, img_v, label) in enumerate(dataloader_train):
            tr_label = torch.cat((tr_label, label.cuda()), 0)
            var_image_a = torch.autograd.Variable(img_a).cuda()
            var_image_h = torch.autograd.Variable(img_h).cuda()
            var_image_v = torch.autograd.Variable(img_v).cuda()
            var_feat, var_clss, mask_h, mask_v = model(var_image_a, var_image_h, var_image_v)
            tr_feat = torch.cat((tr_feat, var_feat.data), 0)
            sys.stdout.write('\r train set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()

    te_label = torch.FloatTensor().cuda()
    te_feat = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (img_a, img_h, img_v, label) in enumerate(dataloader_test):
            te_label = torch.cat((te_label, label.cuda()), 0)
            var_image_a = torch.autograd.Variable(img_a).cuda()
            var_image_h = torch.autograd.Variable(img_h).cuda()
            var_image_v = torch.autograd.Variable(img_v).cuda()
            var_feat, var_clss, mask_h, mask_v = model(var_image_a, var_image_h, var_image_v)
            te_feat = torch.cat((te_feat, var_feat.data), 0)
            sys.stdout.write('\r test set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()

    print('********************Retrieval Performance!********************')
    sim_mat = cosine_similarity(te_feat.cpu().numpy(), tr_feat.cpu().numpy())
    te_label = te_label.cpu().numpy()
    tr_label = tr_label.cpu().numpy()

    for topk in [10]:#[5,10,20,50]:
        mHRs = {0:[], 1:[], 2:[], 3:[], 4:[]} #Hit Ratio
        mHRs_avg = []
        mAPs = {0:[], 1:[], 2:[], 3:[], 4:[]} #mean average precision
        mAPs_avg = []
        #NDCG: lack of ground truth ranking labels
        for i in range(sim_mat.shape[0]):
            idxs, vals = zip(*heapq.nlargest(topk, enumerate(sim_mat[i,:].tolist()), key=lambda x:x[1]))
            num_pos = 0
            rank_pos = 0
            mAP = []
            te_idx = np.where(te_label[i,:]==1)[0][0]
            for j in idxs:
                rank_pos = rank_pos + 1
                tr_idx = np.where(tr_label[j,:]==1)[0][0]  
                if tr_idx == te_idx:  #hit
                    num_pos = num_pos +1
                    mAP.append(num_pos/rank_pos)
                else:
                    mAP.append(0)
            if len(mAP) > 0:
                mAPs[te_idx].append(np.mean(mAP))
                mAPs_avg.append(np.mean(mAP))
            else:
                mAPs[te_idx].append(0)
                mAPs_avg.append(0)
            mHRs[te_idx].append(num_pos/rank_pos)
            mHRs_avg.append(num_pos/rank_pos)
            sys.stdout.write('\r test set process: = {}'.format(i+1))
            sys.stdout.flush()

        #Hit ratio
        for i in range(N_CLASSES):
            logger.info('MANet mHR of {} is {:.4f}'.format(CLASS_NAMES[i], np.mean(mHRs[i])))
        logger.info("MANet Average mHR@{}={:.4f}".format(topk, np.mean(mHRs_avg)))
        #average precision
        for i in range(N_CLASSES):
            logger.info('MANet mAP of {} is {:.4f}'.format(CLASS_NAMES[i], np.mean(mAPs[i])))
        logger.info("MANet Average mAP@{}={:.4f}".format(topk, np.mean(mAPs_avg)))
        #NDCG: normalized discounted cumulative gain

def main():
    Train() #for training
    Test() #for test

if __name__ == '__main__':
    main()
