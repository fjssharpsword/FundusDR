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
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
#self-defined
from config import *
from utils.logger import get_logger
from datasets.KaggleDR import get_train_dataloader, get_validation_dataloader, get_test_dataloader
from nets.SANet import SANet, ContrastiveLoss

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
        model = SANet(num_classes=5).cuda()
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
    criterion = nn.BCELoss().cuda() #ContrastiveLoss().cuda()
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
                _, var_output = model(var_image)
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
        model = SANet(num_classes=5).cuda()
        CKPT_PATH = config['CKPT_PATH'] + args.model + '_best_model_v1.0.pkl'
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
            var_feat, _ = model(var_image)
            tr_feat = torch.cat((tr_feat, var_feat.data), 0)
            sys.stdout.write('\r train set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()

    te_label = torch.FloatTensor().cuda()
    te_feat = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(dataloader_test):
            te_label = torch.cat((te_label, label.cuda()), 0)
            var_image = torch.autograd.Variable(image).cuda()
            var_feat, _ = model(var_image)
            te_feat = torch.cat((te_feat, var_feat.data), 0)
            sys.stdout.write('\r test set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()

    print('********************Retrieval Performance!********************')
    sim_mat = cosine_similarity(te_feat.cpu().numpy(), tr_feat.cpu().numpy())
    te_label = te_label.cpu().numpy()
    tr_label = tr_label.cpu().numpy()

    for topk in [10]: #[5,10,20,50]:
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
            logger.info('SANet mHR of {} is {:.4f}'.format(CLASS_NAMES[i], np.mean(mHRs[i])))
        logger.info("SANet Average mHR@{}={:.4f}".format(topk, np.mean(mHRs_avg)))
        #average precision
        for i in range(N_CLASSES):
            logger.info('SANet mAP of {} is {:.4f}'.format(CLASS_NAMES[i], np.mean(mAPs[i])))
        logger.info("SANet Average mAP@{}={:.4f}".format(topk, np.mean(mAPs_avg)))
        #NDCG: normalized discounted cumulative gain

def VisQuery():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    print('********************load data succeed!********************')

    print('********************load model********************')
    if args.model == 'SANet':
        model = SANet(num_classes=5).cuda()
        CKPT_PATH = config['CKPT_PATH'] + args.model + '_best_model_v1.0.pkl'
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
    name_list = []
    with torch.autograd.no_grad():
        for batch_idx, (image, label, name) in enumerate(dataloader_train):
            tr_label = torch.cat((tr_label, label.cuda()), 0)
            var_image = torch.autograd.Variable(image).cuda()
            var_feat, _ = model(var_image)
            tr_feat = torch.cat((tr_feat, var_feat.data), 0)
            name_list.extend(name)
            sys.stdout.write('\r train set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()

    print('********************Extract feature of query image!********************')
 
    sim_mat = cosine_similarity(tr_feat.data.cpu().numpy(), tr_feat.cpu().numpy())
    tr_label = tr_label.cpu().numpy()
    lbls = [1,2,3,4] #mild DR, moderate DR, severe DR, proliferative DR
    for lbl in lbls:
        indicator = 0
        for i in range(sim_mat.shape[0]):
            if indicator ==1: break
            if np.where(tr_label[i,:]==1)[0][0]==lbl:  
                idxs, vals = zip(*heapq.nlargest(6, enumerate(sim_mat[i,:].tolist()), key=lambda x:x[1]))
                if tr_label[list(idxs)][:,lbl].sum()>4:
                    for j in idxs:
                        print('query image={} and label={}: image={} and label={}'.format(name_list[i], lbl, name_list[j], np.where(tr_label[j,:]==1)[0][0]))
                        indicator = 1
    """
    query_img_path = '/data/pycode/FundusDR/imgs/IDRiD_085_1.jpg'
    transform_seq = transforms.Compose([
        transforms.Resize((config['TRAN_SIZE'], config['TRAN_SIZE'])),
        transforms.ToTensor() #to tesnor [0,1]
    ])
    query_img = transform_seq(Image.open(query_img_path).convert('RGB'))
    var_image = torch.autograd.Variable(torch.unsqueeze(query_img,0)).cuda()
    query_feat, _ = model(var_image)
    sim_mat = cosine_similarity(query_feat.data.cpu().numpy(), tr_feat.cpu().numpy())
    tr_label = tr_label.cpu().numpy()
    idxs, vals = zip(*heapq.nlargest(10000, enumerate(sim_mat[0,:].tolist()), key=lambda x:x[1]))
    for i in idxs:
        if np.where(tr_label[i,:]==1)[0][0]==1:
            print(name_list[i])
    """

def VisMap():
    print('********************load model********************')
    if args.model == 'SANet':
        model = SANet(num_classes=N_CLASSES).cuda()
        CKPT_PATH = config['CKPT_PATH'] + args.model + '_best_model_v1.0.pkl'
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

    query_img_path = '/data/pycode/FundusDR/imgs/IDRiD_46.jpg' #IDRiD_53.jpg  IDRiD_09.jpg
    transform_seq = transforms.Compose([
        transforms.Resize((config['TRAN_SIZE'], config['TRAN_SIZE'])),
        transforms.ToTensor() #to tesnor [0,1]
    ])

    query_img = Image.open(query_img_path).convert('RGB')
    var_image = torch.autograd.Variable(torch.unsqueeze(transform_seq(query_img),0)).cuda()
    x, out, map_bb, map_sa = model(var_image)

    #query_img = query_img.permute(1, 2, 0).numpy()
    #query_img = np.uint8(255 * query_img)
    query_img =  np.asarray(query_img)
    width, height = query_img.shape[0],query_img.shape[1]

    heat_map = map_sa.data.cpu().squeeze(0).sum(dim=0) #1024*8*8
    heat_map = heat_map.squeeze(0).numpy()#.permute(1, 2, 0)
    heat_map = heat_map - heat_map.min()
    heat_map = heat_map / heat_map.max() 
    #plot heat map
    #sns_temp = sns.heatmap(heat_map)
    #sns_temp = sns_temp.get_figure()
    #sns_temp.savefig('/data/pycode/FundusDR/imgs/IDRiD_46_sns.jpg', dpi = 400)
    #resize and convert L to RGB
    heat_map = cv2.resize(heat_map,(height, width))
    heat_map = np.uint8(heat_map * 255.0)
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET) #L to RGB
    
    #weight
    #mask_ma = os.path.join('/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TrainingSet/Microaneurysms/IDRiD_53_MA.tif')
    #mask_ma = Image.open(mask_ma).convert('RGB') #0,255
    #mask_he = os.path.join('/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TrainingSet/Haemorrhages/IDRiD_53_HE.tif')
    #mask_he = Image.open(mask_he).convert('RGB')
    """
    H, W = mask_ma.size
    for w in range(W):
        for h in range(H):
            color_ma = mask_ma.getpixel((h,w))
            color_he = mask_he.getpixel((h,w))
            if color_ma==(255,0,0) or color_he==(255,0,0):
                heat_map[w][h] = [0,0,255] #blue
    """
    #heat_map = heat_map  + np.array(mask_he) + np.array(mask_ma)
    #heat_map = np.where(heat_map>255, np.full_like(heat_map, 255), heat_map) 

    output_img = cv2.addWeighted(query_img, 0.7, heat_map, 0.3, 0)
    plt.imshow(output_img)
    plt.axis('off')
    plt.savefig('/data/pycode/FundusDR/imgs/IDRiD_46_map_sa.jpg')

def main():
    #Train() #for training
    #Test() #for test
    #VisQuery()#for show
    VisMap()

if __name__ == '__main__':
    main()
