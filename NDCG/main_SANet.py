# encoding: utf-8
"""
Training implementation for Mirror Attention
Author: Jason.Fang
Update time: 19/04/2021
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
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
#self-defined
from config import *
from utils.logger import get_logger
from datasets.IDRiDSegHE import get_train_dataloader, get_test_dataloader
from nets.SANet import SANet, RelevanceDegreeLoss
from sklearn.metrics import ndcg_score

#command parameters
parser = argparse.ArgumentParser(description='For FundusDR')
parser.add_argument('--model', type=str, default='SANet', help='SANet')
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
    if args.model == 'SANet':
        model = SANet().cuda()
        CKPT_PATH = config['CKPT_PATH'] + args.model + '_best_model.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model.load_state_dict(checkpoint) #strict=False
            print("=> Loaded well-trained checkpoint from: "+CKPT_PATH)
    else: 
        print('No required model')
        return #over
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_model = lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)
    #define loss function
    criterion = RelevanceDegreeLoss().cuda()#nn.BCELoss().cuda() #ContrastiveLoss().cuda()
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
            for batch_idx, (image, label) in enumerate(dataloader_train):
                optimizer.zero_grad()
                #forward
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                var_output = model(var_image)
                # backward and update parameters
                loss_tensor = criterion.forward(var_output, var_label)
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
            torch.save(model.state_dict(), config['CKPT_PATH'] + args.model + '_best_model.pkl')
            print(' Epoch: {} model has been already save!'.format(epoch + 1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

def Test():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    dataloader_test = get_test_dataloader(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    print('********************load data succeed!********************')

    print('********************load model********************')
    if args.model == 'SANet':
        model = SANet().cuda()
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
        for batch_idx, (image, label) in enumerate(dataloader_train):
            tr_label = torch.cat((tr_label, label.cuda()), 0)
            var_image = torch.autograd.Variable(image).cuda()
            var_feat = model(var_image)
            tr_feat = torch.cat((tr_feat, var_feat.data), 0)
            sys.stdout.write('\r train set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()

    te_label = torch.FloatTensor().cuda()
    te_feat = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(dataloader_test):
            te_label = torch.cat((te_label, label.cuda()), 0)
            var_image = torch.autograd.Variable(image).cuda()
            var_feat = model(var_image)
            te_feat = torch.cat((te_feat, var_feat.data), 0)
            sys.stdout.write('\r test set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()

    print('********************Retrieval Performance!********************')
    sim_mat = cosine_similarity(te_feat.cpu().numpy(), tr_feat.cpu().numpy())
    te_label = te_label.cpu().numpy()
    tr_label = tr_label.cpu().numpy()

    for topk in [10]: #[5,10,20,50]:
        mHRs_avg = []
        mAPs_avg = []
        NDCG_avg = []
        #NDCG: lack of ground truth ranking labels
        for i in range(sim_mat.shape[0]):
            idxs, vals = zip(*heapq.nlargest(topk, enumerate(sim_mat[i,:].tolist()), key=lambda x:x[1]))
            num_pos = 0
            rank_pos = 0
            mAP = []
            te_idx = te_label[i,:][0]
            for j in idxs:
                rank_pos = rank_pos + 1
                tr_idx = tr_label[j,:][0]
                if abs(tr_idx - te_idx) < np.mean(tr_label):  #hit
                    num_pos = num_pos +1
                    mAP.append(num_pos/rank_pos)
                else:
                    mAP.append(0)
            if len(mAP) > 0:
                mAPs_avg.append(np.mean(mAP))
            else:
                mAPs_avg.append(0)
            mHRs_avg.append(num_pos/rank_pos)

            #calculate NDCG
            idxs = np.array(idxs)#tuple -> array
            pd_score = tr_label[idxs].transpose(1,0)
            gt_score = abs(np.sort(-tr_label[idxs],axis=0)).transpose(1,0)
            NDCG_avg.append(ndcg_score(gt_score, pd_score))
            sys.stdout.write('\r test set process: = {}'.format(i+1))
            sys.stdout.flush()

        #Hit ratio
        logger.info("SANet Average mHR@{}={:.4f}".format(topk, np.mean(mHRs_avg)))
        #average precision
        logger.info("SANet Average mAP@{}={:.4f}".format(topk, np.mean(mAPs_avg)))
        #NDCG: normalized discounted cumulative gain
        logger.info("SANet Average mNDCG@{}={:.4f}".format(topk, np.mean(NDCG_avg)))


def main():
    #Train() #for training
    #Test() #for test
    print(ndcg_score([[5,4,0,0,1,0,3,0,2,5]], [[5,5,4,3,2,1,0,0,0,0]]))

if __name__ == '__main__':
    main()
