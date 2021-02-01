# encoding: utf-8
"""
Training implementation
Author: Jason.Fang
Update: Ming Zeng
Update time: 01/02/2021
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
from datasets.KaggleDR_SOLAR import get_train_dataloader, get_validation_dataloader, get_test_dataloader
from utils.Evaluation import compute_AUCs, compute_ROCCurve, compute_IoUs
from nets.AENet import AENet, CircleLoss
from solar_global.networks.imageretrievalnet import init_network,SOLAR_Global_Retrieval
from solar_global.networks.networks import ResNetSOAs
from solar_global.layers.loss import TripletLoss,SOSLoss

#command parameters
parser = argparse.ArgumentParser(description='For FundusDR')
parser.add_argument('--model', type=str, default='AENet', help='SOLAR')
args = parser.parse_args()

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
logger = get_logger(config['log_path'])

def Train():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=8)
    dataloader_val = get_validation_dataloader(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    params = {}
    model = init_network(params).cuda()  # initialize model

    # model_img = nn.DataParallel(model_img).cuda()  # make model available multi GPU cores training
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_model = lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)

    # torch.backends.cudnn.benchmark = False  # improve train speed slightly
    torch.backends.cudnn.enabled = False

    triplet_loss = TripletLoss(margin=0.7).cuda()
    second_loss = SOSLoss().cuda()

    print('********************load model succeed!********************')

    print('********************begin training!********************')
    AUROC_best = 0.50
    for epoch in range(config['MAX_EPOCHS']):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , config['MAX_EPOCHS']))
        print('-' * 10)
        model.train()  #set model to training mode
        # cls_loss, cir_loss, reg_loss, train_loss = [], [], [], []

        train_loss = []
        with torch.autograd.enable_grad():
            for batch_idx, (image, label) in enumerate(dataloader_train):

                optimizer.zero_grad()
                #forward
                var_image_anchor = torch.autograd.Variable(image[0]).cuda()
                # var_label_anchor = torch.autograd.Variable(label[:,0]).cuda() # var_label = [32,5]
                var_feat_anchor,_= model(var_image_anchor) # var_feat = [32,2048]

                var_image_pos = torch.autograd.Variable(image[1]).cuda()
                # var_label_pos = torch.autograd.Variable(label[:,1]).cuda()  # var_label = [32,5]
                var_feat_pos,_ = model(var_image_pos)  # var_feat = [32,2048]

                var_image_neg = torch.autograd.Variable(image[2]).cuda()
                # var_label_neg = torch.autograd.Variable(label[:,2]).cuda()  # var_label = [32,5]
                var_feat_neg,_ = model(var_image_neg)  # var_feat = [32,2048]

                loss_tri = triplet_loss.forward(var_feat_anchor, var_feat_pos, var_feat_neg)
                # loss_tri.backward(retain_graph=True)
                loss_second = second_loss.forward(var_feat_anchor, var_feat_pos, var_feat_neg)
                # loss_second.backward()
                loss = loss_tri + 5 * loss_second
                loss.backward()
                optimizer.step()
                loss_tensor = loss
                train_loss.append(loss_tensor.item())
                #print([x.grad for x in optimizer.param_groups[0]['params']])
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'\
                                .format(epoch+1, batch_idx+1,  float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()



        lr_scheduler_model.step()  #about lr and gamma
        print("\r Eopch: %5d train loss = %.6f" \
            % (epoch + 1, np.mean(train_loss)))

        model.eval() #turn to test mode
        val_loss = []

        gt = torch.FloatTensor().cuda()
        pred = torch.FloatTensor().cuda()

        with torch.autograd.no_grad():
            for batch_idx, (image, label) in enumerate(dataloader_val):
                gt = torch.cat((gt, label[:,0].cuda()), 0)
                #forward
                var_image_anchor = torch.autograd.Variable(image[0]).cuda()
                # var_label_anchor = torch.autograd.Variable(label[:,0]).cuda() # var_label = [32,5]
                var_feat_anchor, var_output= model(var_image_anchor)  # var_feat = [32,2048]

                var_image_pos = torch.autograd.Variable(image[1]).cuda()
                # var_label_pos = torch.autograd.Variable(label[:,1]).cuda()  # var_label = [32,5]
                var_feat_pos, _ = model(var_image_pos)  # var_feat = [32,2048]

                var_image_neg = torch.autograd.Variable(image[2]).cuda()
                # var_label_neg = torch.autograd.Variable(label[:,2]).cuda()  # var_label = [32,5]
                var_feat_neg, _ = model(var_image_neg)  # var_feat = [32,2048]

                pred = torch.cat((pred, var_output.data), 0)

                #backward and update parameters

                loss_tri = triplet_loss.forward(var_feat_anchor, var_feat_pos, var_feat_neg)
                loss_second = second_loss.forward(var_feat_anchor, var_feat_pos, var_feat_neg)
                loss = loss_tri + 5 * loss_second

                loss_tensor = loss

                val_loss.append(loss_tensor.item())

                sys.stdout.write('\r Epoch: {} / Step: {} : validation loss = {}'\
                                .format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()
        #evaluation
        AUROC_avg = np.array(compute_AUCs(gt, pred)).mean()
        logger.info("\r Eopch: %5d validation loss = %.4f, Average AUROC =%.4f"
                     % (epoch + 1, np.mean(val_loss), AUROC_avg))
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

    params = {}
    model = init_network(params).cuda()  # initialize model

    CKPT_PATH = config['CKPT_PATH'] + args.model + '/best_model.pkl'
    checkpoint = torch.load(CKPT_PATH)
    model.load_state_dict(checkpoint)  # strict=False
    print("=> loaded model checkpoint: " + CKPT_PATH)


    # torch.backends.cudnn.benchmark = True  # improve train speed slightly
    torch.backends.cudnn.enabled = False
    model.eval()#turn to test mode
    print('******************** load model succeed!********************')

    print('********************Build feature database!********************')
    tr_label = torch.FloatTensor().cuda()
    tr_feat = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(dataloader_train):
            tr_label = torch.cat((tr_label, label[:,0].cuda()), 0)
            var_image = torch.autograd.Variable(image[0]).cuda()
            var_feat, _ = model(var_image)
            tr_feat = torch.cat((tr_feat, var_feat.data), 0)
            sys.stdout.write('\r train set process: = {}'.format(batch_idx+1))
            sys.stdout.flush()

    te_label = torch.FloatTensor().cuda()
    te_feat = torch.FloatTensor().cuda()
    te_pred = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(dataloader_test):
            te_label = torch.cat((te_label, label[:,0].cuda()), 0)
            var_image = torch.autograd.Variable(image[0]).cuda()
            var_feat, var_output= model(var_image)
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
