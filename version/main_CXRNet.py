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
 
#self-defined
from config import *
from datasets.KaggleDR import get_train_dataloader, get_validation_dataloader, get_test_dataloader
from utils.Evaluation import compute_AUCs, compute_ROCCurve, compute_IoUs
from nets.CXRNet import ImageClassifier, RegionClassifier, FusionClassifier

#command parameters
parser = argparse.ArgumentParser(description='For FundusDR')
parser.add_argument('--model', type=str, default='CXRNet', help='CXRNet')
args = parser.parse_args()

#config
os.environ['CUDA_VISIBLE_DEVICES'] = config['CUDA_VISIBLE_DEVICES']

def Train():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=8)
    dataloader_val = get_validation_dataloader(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    #dataloader_train, dataloader_val = get_train_dataloader_full(batch_size=BATCH_SIZE, shuffle=True, num_workers=8, split_ratio=0.1)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'CXRNet':
        model_img = ImageClassifier(num_classes=N_CLASSES, is_pre_trained=True).cuda()#initialize model 
        #model_img = nn.DataParallel(model_img).cuda()  # make model available multi GPU cores training
        optimizer_img = optim.Adam(model_img.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        lr_scheduler_img = lr_scheduler.StepLR(optimizer_img, step_size = 10, gamma = 1)

        model_roi = RegionClassifier(num_classes=N_CLASSES, is_pre_trained=True).cuda()
        #model_roi = nn.DataParallel(model_roi).cuda()
        optimizer_roi = optim.Adam(model_roi.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        lr_scheduler_roi = lr_scheduler.StepLR(optimizer_roi, step_size = 10, gamma = 1)

        model_fusion = FusionClassifier(input_size=2048, output_size=N_CLASSES).cuda()
        #model_fusion = nn.DataParallel(model_fusion).cuda()
        optimizer_fusion = optim.Adam(model_fusion.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        lr_scheduler_fusion = lr_scheduler.StepLR(optimizer_fusion, step_size = 10, gamma = 1)
    else: 
        print('No required model')
        return #over

    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    bce_criterion = nn.BCELoss() #define binary cross-entropy loss
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    AUROC_best = 0.50
    for epoch in range(config['MAX_EPOCHS']):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , config['MAX_EPOCHS']))
        print('-' * 10)
        model_img.train()  #set model to training mode
        model_roi.train()
        model_fusion.train()
        train_loss = []
        with torch.autograd.enable_grad():
            for batch_idx, (image, label) in enumerate(dataloader_train):
                optimizer_img.zero_grad()
                optimizer_roi.zero_grad() 
                optimizer_fusion.zero_grad() 
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                #image-level
                fc_fea_img, out_img = model_img(var_image)#forward
                loss_img = bce_criterion(out_img, var_label)
                loss_img.backward()
                optimizer_img.step()
                #ROI-level
                fc_fea_roi, out_roi = model_roi(var_image)
                loss_roi = bce_criterion(out_roi, var_label) 
                loss_roi.backward()
                optimizer_roi.step()
                #Fusion
                fc_fea_fusion = torch.cat((fc_fea_img,fc_fea_roi), 1)
                var_fusion = torch.autograd.Variable(fc_fea_fusion).cuda()
                out_fusion = model_fusion(var_fusion)
                loss_fusion = bce_criterion(out_fusion, var_label) 
                loss_fusion.backward()
                optimizer_fusion.step() 
                #backward and update parameters 
                loss_tensor = loss_img + loss_roi + loss_fusion 
                train_loss.append(loss_tensor.item())
                #print([x.grad for x in optimizer.param_groups[0]['params']])
                sys.stdout.write('\r Epoch: {} / Step: {} : image loss ={}, roi loss ={}, fusion loss = {}, train loss = {}'
                                .format(epoch+1, batch_idx+1, float('%0.6f'%loss_img.item()), float('%0.6f'%loss_roi.item()),
                                float('%0.6f'%loss_fusion.item()), float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()        
        lr_scheduler_img.step()  #about lr and gamma
        lr_scheduler_roi.step()
        lr_scheduler_fusion.step()
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(train_loss))) 

        model_img.eval() #turn to test mode
        model_roi.eval()
        model_fusion.eval()
        val_loss = []
        gt = torch.FloatTensor().cuda()
        pred_img = torch.FloatTensor().cuda()
        pred_roi = torch.FloatTensor().cuda()
        pred_fusion = torch.FloatTensor().cuda()
        with torch.autograd.no_grad():
            for batch_idx, (image, label) in enumerate(dataloader_val):
                gt = torch.cat((gt, label.cuda()), 0)
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                #image-level
                fc_fea_img, out_img = model_img(var_image)#forward
                loss_img = bce_criterion(out_img, var_label) 
                pred_img = torch.cat((pred_img, out_img.data), 0)
                #ROI-level
                fc_fea_roi, out_roi = model_roi(var_image)
                loss_roi = bce_criterion(out_roi, var_label) 
                pred_roi = torch.cat((pred_roi, out_roi.data), 0)
                #Fusion
                fc_fea_fusion = torch.cat((fc_fea_img,fc_fea_roi), 1)
                var_fusion = torch.autograd.Variable(fc_fea_fusion).cuda()
                out_fusion = model_fusion(var_fusion)
                loss_fusion = bce_criterion(out_fusion, var_label) 
                pred_fusion = torch.cat((pred_fusion, out_fusion.data), 0)
                #loss
                loss_tensor = loss_img + loss_roi + loss_fusion
                val_loss.append(loss_tensor.item())
                sys.stdout.write('\r Epoch: {} / Step: {} : image loss ={}, roi loss ={}, fusion loss = {}, train loss = {}'
                                .format(epoch+1, batch_idx+1, float('%0.6f'%loss_img.item()), float('%0.6f'%loss_roi.item()),
                                float('%0.6f'%loss_fusion.item()), float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()
        #evaluation       
        AUROCs_img = np.array(compute_AUCs(gt, pred_img)).mean()
        AUROCs_roi = np.array(compute_AUCs(gt, pred_roi)).mean()
        AUROCs_fusion = np.array(compute_AUCs(gt, pred_fusion)).mean()
        print("\r Eopch: %5d validation loss = %.6f, Validataion AUROC image=%.4f roi=%.4f fusion=%.4f" 
              % (epoch + 1, np.mean(val_loss), AUROCs_img, AUROCs_roi, AUROCs_fusion)) 
        #save checkpoint
        if AUROC_best < AUROCs_fusion:
            AUROC_best = AUROCs_fusion
            #torch.save(model.module.state_dict(), CKPT_PATH)
            torch.save(model_img.state_dict(), config['CKPT_PATH']+ args.model +'/img_model.pkl') #Saving torch.nn.DataParallel Models
            torch.save(model_roi.state_dict(), config['CKPT_PATH']+ args.model +'/roi_model.pkl')
            torch.save(model_fusion.state_dict(), config['CKPT_PATH']+ args.model +'/fusion_model.pkl')
            print(' Epoch: {} model has been already save!'.format(epoch+1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

def Test():
    print('********************load data********************')
    dataloader_test = get_test_dataloader(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'CXRNet':
        model_img = ImageClassifier(num_classes=N_CLASSES, is_pre_trained=True).cuda()
        CKPT_PATH = config['CKPT_PATH']+ args.model +'/img_model.pkl'
        checkpoint = torch.load(CKPT_PATH)
        model_img.load_state_dict(checkpoint) #strict=False
        print("=> loaded Image model checkpoint: "+CKPT_PATH)

        model_roi = RegionClassifier(num_classes=N_CLASSES, is_pre_trained=True).cuda()
        CKPT_PATH = config['CKPT_PATH']+ args.model +'/roi_model.pkl'
        checkpoint = torch.load(CKPT_PATH)
        model_roi.load_state_dict(checkpoint) #strict=False
        print("=> loaded ROI model checkpoint: "+CKPT_PATH)

        model_fusion = FusionClassifier(input_size=2048, output_size=N_CLASSES).cuda()
        CKPT_PATH = config['CKPT_PATH']+ args.model +'/fusion_model.pkl'
        checkpoint = torch.load(CKPT_PATH)
        model_fusion.load_state_dict(checkpoint) #strict=False
        print("=> loaded Fusion model checkpoint: "+CKPT_PATH)
    else: 
        print('No required model')
        return #over
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    print('******************** load model succeed!********************')

    print('******* begin testing!*********')
    gt = torch.FloatTensor().cuda()
    pred_img = torch.FloatTensor().cuda()
    pred_roi = torch.FloatTensor().cuda()
    pred_fusion = torch.FloatTensor().cuda()
    # switch to evaluate mode
    model_img.eval() #turn to test mode
    model_roi.eval()
    model_fusion.eval()
    cudnn.benchmark = True
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(dataloader_test):
            gt = torch.cat((gt, label.cuda()), 0)
            var_image = torch.autograd.Variable(image).cuda()
            #image-level
            fc_fea_img, out_img = model_img(var_image)#forward
            pred_img = torch.cat((pred_img, out_img.data), 0)
            #ROI-level
            fc_fea_roi, out_roi = model_roi(var_image)
            pred_roi = torch.cat((pred_roi, out_roi.data), 0)
            #Fusion
            fc_fea_fusion = torch.cat((fc_fea_img,fc_fea_roi), 1)
            var_fusion = torch.autograd.Variable(fc_fea_fusion).cuda()
            out_fusion = model_fusion(var_fusion)
            pred_fusion = torch.cat((pred_fusion, out_fusion.data), 0)
            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()

    #for evaluation
    AUROC_img = compute_AUCs(gt, pred_img)
    AUROC_avg = np.array(AUROC_img).mean()
    for i in range(N_CLASSES):
        print('The AUROC of {} is {:.4f}'.format(CLASS_NAMES[i], AUROC_img[i]))
    print('The average AUROC is {:.4f}'.format(AUROC_avg))

    AUROC_roi = compute_AUCs(gt, pred_roi)
    AUROC_avg = np.array(AUROC_roi).mean()
    for i in range(N_CLASSES):
        print('The AUROC of {} is {:.4f}'.format(CLASS_NAMES[i], AUROC_roi[i]))
    print('The average AUROC is {:.4f}'.format(AUROC_avg))

    AUROC_fusion = compute_AUCs(gt, pred_fusion)
    AUROC_avg = np.array(AUROC_fusion).mean()
    for i in range(N_CLASSES):
        print('The AUROC of {} is {:.4f}'.format(CLASS_NAMES[i], AUROC_fusion[i]))
    print('The average AUROC is {:.4f}'.format(AUROC_avg))

    #Evaluating the threshold of prediction
    thresholds = compute_ROCCurve(gt, pred_fusion)
    print(thresholds)


def main():
    Train() #for training
    Test() #for test

if __name__ == '__main__':
    main()