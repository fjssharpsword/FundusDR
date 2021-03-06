# encoding: utf-8
"""
Training implementation
Author: Jason.Fang
Update time: 08/12/2020
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
#self-defined
from config import *
from utils.logger import get_logger
from datasets.KaggleDR import get_train_dataloader, get_validation_dataloader, get_test_dataloader
from utils.Evaluation import compute_AUCs, compute_ROCCurve, compute_IoUs
from nets.AENet import AENet, CircleLoss

#command parameters
parser = argparse.ArgumentParser(description='For FundusDR')
parser.add_argument('--model', type=str, default='AENet', help='AENet')
args = parser.parse_args()

#config
os.environ['CUDA_VISIBLE_DEVICES'] = config['CUDA_VISIBLE_DEVICES']
logger = get_logger(config['log_path'])

def Train():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=8)
    dataloader_val = get_validation_dataloader(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'AENet':
        model = AENet(num_classes=N_CLASSES).cuda()#initialize model 
        #model_img = nn.DataParallel(model_img).cuda()  # make model available multi GPU cores training
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        lr_scheduler_model = lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 1)
    else: 
        print('No required model')
        return #over

    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    bce_criterion = nn.BCELoss() #define binary cross-entropy loss
    mse_criterion = nn.MSELoss() #define regression loss
    cir_criterion = CircleLoss() #circle loss for ranking
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    AUROC_best = 0.50
    for epoch in range(config['MAX_EPOCHS']):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , config['MAX_EPOCHS']))
        print('-' * 10)
        model.train()  #set model to training mode
        cls_loss, cir_loss, reg_loss, train_loss = [], [], [], []
        with torch.autograd.enable_grad():
            for batch_idx, (image, label) in enumerate(dataloader_train):
                optimizer.zero_grad()
                #forward
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                var_feat, var_output, var_mask = model(var_image)
                #backward and update parameters
                loss_cls = bce_criterion(var_output, var_label)
                loss_cls.backward(retain_graph=True)
                loss_cir = cir_criterion(var_feat, var_label)
                loss_cir.backward(retain_graph=True)
                loss_reg = mse_criterion(var_mask, var_image)
                loss_reg.backward()
                optimizer.step()
                loss_tensor = loss_cls + loss_cir + loss_reg
                train_loss.append(loss_tensor.item())
                cls_loss.append(loss_cls.item())
                cir_loss.append(loss_cir.item())
                reg_loss.append(loss_reg.item())
                #print([x.grad for x in optimizer.param_groups[0]['params']])
                sys.stdout.write('\r Epoch: {} / Step: {} : classification loss ={}, similarity loss = {}, regression loss ={}, train loss = {}'\
                                .format(epoch+1, batch_idx+1, float('%0.6f'%loss_cls.item()), float('%0.6f'%loss_cir.item()),\
                                        float('%0.6f'%loss_reg.item()), float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()        
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Eopch: %5d train loss = %.6f, classificaiton loss =%.6f, similarity loss =%.6f, regression loss=%.6f" \
            % (epoch + 1, np.mean(train_loss), np.mean(cir_loss), np.mean(cls_loss), np.mean(reg_loss))) 

        model.eval() #turn to test mode
        cls_loss, cir_loss, reg_loss, val_loss = [], [], [], []
        gt = torch.FloatTensor().cuda()
        pred = torch.FloatTensor().cuda()
        with torch.autograd.no_grad():
            for batch_idx, (image, label) in enumerate(dataloader_val):
                gt = torch.cat((gt, label.cuda()), 0)
                #forward
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                var_feat, var_output, var_mask = model(var_image)
                pred = torch.cat((pred, var_output.data), 0) 
                #backward and update parameters
                loss_cls = bce_criterion(var_output, var_label)
                loss_cir = cir_criterion(var_feat, var_label)
                loss_reg = mse_criterion(var_mask, var_image)
                loss_tensor = loss_cls + loss_cir + loss_reg
                val_loss.append(loss_tensor.item())
                cls_loss.append(loss_cls.item())
                cir_loss.append(loss_cir.item())
                reg_loss.append(loss_reg.item())
                sys.stdout.write('\r Epoch: {} / Step: {} : classification loss ={}, similarity loss = {}, regression loss ={}, validation loss = {}'\
                                .format(epoch+1, batch_idx+1, float('%0.6f'%loss_cls.item()), float('%0.6f'%loss_cir.item()),\
                                        float('%0.6f'%loss_reg.item()), float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()
        #evaluation       
        AUROC_avg = np.array(compute_AUCs(gt, pred)).mean()
        logger.info("\r Eopch: %5d validation loss = %.4f, classification loss = %.4f,  similarity loss = %.4f, regression loss =%.4f, Average AUROC =%.4f" 
                     % (epoch + 1, np.mean(val_loss), np.mean(cls_loss), np.mean(cir_loss), np.mean(reg_loss), AUROC_avg))
        #save checkpoint
        if AUROC_best < AUROC_avg:
            AUROC_best = AUROC_avg
            #torch.save(model.module.state_dict(), CKPT_PATH)#Saving torch.nn.DataParallel Models
            torch.save(model.state_dict(), config['CKPT_PATH']+ args.model +'/best_model.pkl') 
            print(' Epoch: {} model has been already save!'.format(epoch+1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

def Test():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    dataloader_test = get_test_dataloader(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'AENet':
        model = AENet(num_classes=N_CLASSES).cuda()
        CKPT_PATH = config['CKPT_PATH']+ args.model +'/best_model.pkl'
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> loaded model checkpoint: "+CKPT_PATH)
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
            var_feat, _, _ = model(var_image)
            tr_feat = torch.cat((tr_feat, var_feat.data), 0)
            sys.stdout.write('\r train set process: = {}'.format(batch_idx+1))
            sys.stdout.flush()

    te_label = torch.FloatTensor().cuda()
    te_feat = torch.FloatTensor().cuda()
    te_pred = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(dataloader_test):
            te_label = torch.cat((te_label, label.cuda()), 0)
            var_image = torch.autograd.Variable(image).cuda()
            var_feat, var_output, _ = model(var_image)
            te_feat = torch.cat((te_feat, var_feat.data), 0)
            te_pred = torch.cat((te_pred, var_output.data), 0)
            sys.stdout.write('\r test set process: = {}'.format(batch_idx+1))
            sys.stdout.flush()

    print('********************Classification Performance!********************')
    AUROC_class = compute_AUCs(te_label, te_pred)
    AUROC_avg = np.array(AUROC_class).mean()
    for i in range(N_CLASSES):
        print('The AUROC of {} is {:.4f}'.format(CLASS_NAMES[i], AUROC_class[i]))
    print('The average AUROC is {:.4f}'.format(AUROC_avg))

    print('********************Retrieval Performance!********************')
    sim_mat = cosine_similarity(te_feat.cpu().numpy(), tr_feat.cpu().numpy())
    te_label = te_label.cpu().numpy()
    tr_label = tr_label.cpu().numpy()
    for topk in [5,10,20,50]:
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
            for j in idxs:
                rank_pos = rank_pos + 1
                tr_idx = np.where(tr_label[j,:]==1)[0][0]
                te_idx = np.where(te_label[i,:]==1)[0][0]
                if tr_idx == te_idx:  #hit
                    num_pos = num_pos +1
                    mAP.append(num_pos/rank_pos) 
            if len(mAP) > 0: 
                mAPs[te_idx].append(np.mean(mAP))
                mAPs_avg.append(np.mean(mAP))
            #else:
            #    mAPs[te_idx].append(0)
            #    mAPs_avg.append(0)
            mHRs[te_idx].append(num_pos/rank_pos)
            mHRs_avg.append(num_pos/rank_pos)
            sys.stdout.write('\r test set process: = {}'.format(i+1))
            sys.stdout.flush()

        #Hit ratio
        for i in range(N_CLASSES):
            print('The mHR of {} is {:.4f}'.format(CLASS_NAMES[i], np.mean(mHRs[i])))
        print("Average mHR@{}={:.4f}".format(topk, np.mean(mHRs_avg)))
        #average precision
        for i in range(N_CLASSES):
            print('The mAP of {} is {:.4f}'.format(CLASS_NAMES[i], np.mean(mAPs[i])))
        print("Average mAP@{}={:.4f}".format(topk, np.mean(mAPs_avg)))
        #NDCG: normalized discounted cumulative gain


def main():
    Train() #for training
    Test() #for test

if __name__ == '__main__':
    main()