import argparse
import math
import os
import sys
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms, utils
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import numpy as np
import nets.lpips as lpips
from datasets.fundus_idrid_grading import get_train_dataset_fundus, get_test_dataset_fundus
from nets.model import Generator, Discriminator, TransEnc, CNNEnc
from nets.perceptual import VGGPerceptualLoss
from sklearn.metrics import ndcg_score

def retrivel_per():    
    #load model
    device = "cuda:7"
    transenc = TransEnc(256, 256).to(device)
    cnnsenc = CNNEnc(256, 256).to(device)
    ckpt_path = "/data/pycode/TransGAN/ckpts/149999.pt"
    transenc.load_state_dict(torch.load(ckpt_path, map_location={'cuda:0': 'cuda:7'})["trans"], strict=False)
    transenc.eval()
    cnnsenc.load_state_dict(torch.load(ckpt_path, map_location={'cuda:0': 'cuda:7'})["cnns"], strict=False)
    cnnsenc.eval()

    #load dataset
    train_loader = torch.utils.data.DataLoader(get_train_dataset_fundus(), batch_size=16, shuffle=True,num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(get_test_dataset_fundus(), batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    #retrieval evaluation
    print('********************Build feature for trainset!********************')
    tr_label = torch.FloatTensor()
    tr_feat = torch.FloatTensor()
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(train_loader):
            tr_label = torch.cat((tr_label, label), 0)

            latent_in_trans, latent_in_cnn = transenc(image.to(device)),  cnnsenc(image.to(device))
            var_feat = torch.cat((latent_in_trans, latent_in_cnn), dim=-1)
            #var_feat = cnnsenc(image.to(device))

            tr_feat = torch.cat((tr_feat, var_feat.cpu().data.view(var_feat.shape[0],-1)), 0)
            sys.stdout.write('\r train set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()
    print('********************Extract feature for testset!********************')
    te_label = torch.FloatTensor()
    te_feat = torch.FloatTensor()
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(test_loader):
            te_label = torch.cat((te_label, label), 0)

            latent_in_trans, latent_in_cnn = transenc(image.to(device)),  cnnsenc(image.to(device))
            var_feat = torch.cat((latent_in_trans, latent_in_cnn), dim=-1)
            #var_feat = cnnsenc(image.to(device))

            te_feat = torch.cat((te_feat, var_feat.cpu().data.view(var_feat.shape[0],-1)), 0)
            sys.stdout.write('\r test set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()
    print('********************Retrieval Performance!********************')
    sim_mat = cosine_similarity(te_feat.numpy(), tr_feat.numpy())
    te_label = te_label.numpy()
    tr_label = tr_label.numpy()

    for topk in [5, 10, 20]:
        mHRs = {0: [], 1: [], 2: [], 3: [], 4: []}  # Hit Ratio
        mHRs_avg = []
        mAPs = {0: [], 1: [], 2: [], 3: [], 4: []}  # mean average precision
        mAPs_avg = []
        # NDCG: lack of ground truth ranking labels
        for i in range(sim_mat.shape[0]):
            idxs, vals = zip(*heapq.nlargest(topk, enumerate(sim_mat[i, :].tolist()), key=lambda x: x[1]))
            num_pos = 0
            rank_pos = 0
            mAP = []
            te_idx = np.where(te_label[i, :] == 1)[0][0]
            for j in idxs:
                rank_pos = rank_pos + 1
                tr_idx = np.where(tr_label[j, :] == 1)[0][0]
                if tr_idx == te_idx:  # hit
                    num_pos = num_pos + 1
                    mAP.append(num_pos / rank_pos)
                else:
                    mAP.append(0)
            if len(mAP) > 0:
                mAPs[te_idx].append(np.mean(mAP))
                mAPs_avg.append(np.mean(mAP))
            else:
                mAPs[te_idx].append(0)
                mAPs_avg.append(0)
            mHRs[te_idx].append(num_pos / rank_pos)
            mHRs_avg.append(num_pos / rank_pos)
            sys.stdout.write('\r test set process: = {}'.format(i + 1))
            sys.stdout.flush()

        CLASS_NAMES = ['Normal', "Mild NPDR", 'Moderate NPDR', 'Severe NPDR', 'PDR']
        # Hit ratio
        for i in range(len(CLASS_NAMES)):
            print('Fundus mHR of {} is {:.4f}'.format(CLASS_NAMES[i], np.mean(mHRs[i])))
        print("Fundus Average mHR@{}={:.4f}".format(topk, np.mean(mHRs_avg)))
        # average precision
        for i in range(len(CLASS_NAMES)):
            print('Fundus mAP of {} is {:.4f}'.format(CLASS_NAMES[i], np.mean(mAPs[i])))
        print("Fundus Average mAP@{}={:.4f}".format(topk, np.mean(mAPs_avg)))


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

def recover_fundus():
    #load model
    device = "cuda:6"
    transenc = TransEnc(256, 256).to(device)
    cnnsenc = CNNEnc(256, 256).to(device)
    ckpt_path = "/data/pycode/TransGAN/ckpts/199999.pt"
    transenc.load_state_dict(torch.load(ckpt_path, map_location={'cuda:0': 'cuda:6'})["trans"], strict=False)
    transenc.eval()
    cnnsenc.load_state_dict(torch.load(ckpt_path, map_location={'cuda:0': 'cuda:6'})["cnns"], strict=False)
    cnnsenc.eval()
    g_ema = Generator(256, 512, 8)
    g_ema.load_state_dict(torch.load(ckpt_path, map_location={'cuda:0': 'cuda:6'})["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    #load dataset
    train_loader = torch.utils.data.DataLoader(get_train_dataset_fundus(), batch_size=8, shuffle=True,num_workers=0, pin_memory=True)
    with torch.autograd.no_grad():
        for batch_idx, (imgs, lbls) in enumerate(train_loader):
            imgs = imgs.to(device)
            latent_in_trans, latent_in_cnn = transenc(imgs),  cnnsenc(imgs)
            latent_in = torch.cat((latent_in_trans, latent_in_cnn), dim=-1)

            with torch.autograd.enable_grad():
                latent_in.requires_grad = True
                optimizer = optim.Adam([latent_in], lr=0.1)
                pbar = tqdm(range(10000))
                latent_path = []
                for i in pbar:
                    t = i / 10000
                    lr = get_lr(t, 0.1)
                    optimizer.param_groups[0]["lr"] = lr
            
                    img_gen, _ = g_ema([latent_in], input_is_latent=True)
                    mse_loss = F.mse_loss(img_gen, imgs)
                    optimizer.zero_grad()
                    mse_loss.backward()
                    optimizer.step()
                    if (i + 1) % 100 == 0:
                        latent_path.append(latent_in.detach().clone())

                    pbar.set_description((f"perceptual: {mse_loss.item():.4f}; lr: {lr:.4f}"))

            img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True)
            utils.save_image(
                            torch.cat((imgs.cpu().data, img_gen.cpu().data), 0),
                            f"/data/pycode/TransGAN/logs/recov_fundus.png",
                            nrow=8,
                            normalize=True,
                            range=(-1, 1),
                        )
            break

def perceptual_evaluation():
    #load model
    device = "cuda:7"
    transenc = TransEnc(256, 256).to(device)
    cnnsenc = CNNEnc(256, 256).to(device)
    ckpt_path = "/data/pycode/TransGAN/ckpts/199999.pt"
    transenc.load_state_dict(torch.load(ckpt_path, map_location={'cuda:0': 'cuda:7'})["trans"], strict=False)
    transenc.eval()
    cnnsenc.load_state_dict(torch.load(ckpt_path, map_location={'cuda:0': 'cuda:7'})["cnns"], strict=False)
    cnnsenc.eval()
    vggptloss = VGGPerceptualLoss(matching_loss="mse").to(device)

    #load dataset
    train_loader = torch.utils.data.DataLoader(get_train_dataset_fundus(), batch_size=1, shuffle=True,num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(get_test_dataset_fundus(), batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    #retrieval evaluation
    print('********************Build feature for trainset!********************')
    tr_image = torch.FloatTensor()
    tr_feat = torch.FloatTensor()
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(train_loader):
            tr_image = torch.cat((tr_image, image), 0)

            latent_in_trans, latent_in_cnn = transenc(image.to(device)),  cnnsenc(image.to(device))
            var_feat = torch.cat((latent_in_trans, latent_in_cnn), dim=-1)
            #var_feat = cnnsenc(image.to(device))

            tr_feat = torch.cat((tr_feat, var_feat.cpu().data.view(var_feat.shape[0],-1)), 0)
            sys.stdout.write('\r train set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()
    print('********************Extract feature for testset!********************')
    te_image = torch.FloatTensor()
    te_feat = torch.FloatTensor()
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(test_loader):
            te_image = torch.cat((te_image, image), 0)

            latent_in_trans, latent_in_cnn = transenc(image.to(device)),  cnnsenc(image.to(device))
            var_feat = torch.cat((latent_in_trans, latent_in_cnn), dim=-1)
            #var_feat = cnnsenc(image.to(device))

            te_feat = torch.cat((te_feat, var_feat.cpu().data.view(var_feat.shape[0],-1)), 0)
            sys.stdout.write('\r test set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()
    """
    print('********************Calculate the maximum and minimum perception loss!********************')
    max_p_loss, min_p_loss = 0.0, 256*256 
    with torch.autograd.no_grad():
        for batch_idx, (trimage, trlabel) in enumerate(train_loader):
            for batch_idx, (teimage, telabel) in enumerate(train_loader):
                p_loss = vggptloss(trimage.to(device), teimage.to(device))
                if p_loss>max_p_loss: max_p_loss = p_loss
                if p_loss<min_p_loss: min_p_loss = p_loss
    print("Maximum and Minimmum Perceptions are  {:.4f} and {:.4f}".format(max_p_loss, min_p_loss))
    """
    # max_p_loss=5.5638, min_p_loss=0.0
    print('********************Retrieval Performance!********************')
    sim_mat = cosine_similarity(te_feat.numpy(), tr_feat.numpy())
    tr_image = tr_image.to(device)
    te_image = te_image.to(device)

    for topk in [5, 10, 20]:
        # NDCG: lack of ground truth ranking labels
        perc_seq, NDCG_avg = [], []
        for i in range(sim_mat.shape[0]):
            idxs, vals = zip(*heapq.nlargest(topk, enumerate(sim_mat[i, :].tolist()), key=lambda x: x[1]))
            query_img = te_image[i,:,:,:]
            loss_seq = []
            for j in idxs:
                return_img = tr_image[i,:,:,:]
                p_loss = vggptloss(query_img.unsqueeze(0), return_img.unsqueeze(0))
                #p_loss = (p_loss-min_p_loss)/(max_p_loss-min_p_loss)
                loss_seq.append(p_loss.cpu().data)

            perc_seq.append(np.mean(loss_seq))
            #calculate NDCG
            #idxs = np.arange(topk)#tuple -> array
            #loss_seq = np.array([loss_seq])
            #pd_loss = loss_seq.transpose(1,0)
            #gt_loss = abs(np.sort(-loss_seq,axis=0)).transpose(1,0)
            #NDCG_avg.append(ndcg_score(gt_loss, pd_loss))

            sys.stdout.write('\r test set process: = {}'.format(i+1))
            sys.stdout.flush()

        # average perceptual
        print("Fundus Average Perception@{}={:.4f}".format(topk, np.mean(perc_seq)))
        #print("Fundus Average NDCG@{}={:.4f}".format(topk, np.mean(NDCG_avg)))

if __name__ == "__main__":

    #retrivel_per()
    #recover_fundus()
    perceptual_evaluation()
    #python proj_fundus.py