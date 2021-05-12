# encoding: utf-8
"""
Training implementation
Author: Jason.Fang
Update time: 16/03/2021
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
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
from skimage.measure import label
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.ndimage import binary_dilation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import PIL.ImageOps
import torchvision.transforms as transforms

#Visualize the mirror images
def genMirrorImages():
    """
    transform_tensor = transforms.Compose([transforms.ToTensor()]) #to tesnor [0,1]
    img_path = '/data/pycode/FundusDR/imgs/IDRiD_085.jpg'
    image = Image.open(img_path).convert('RGB')
    image = transform_tensor(image)
    avg_out = torch.mean(image, dim=0, keepdim=True)
    avg_img = transforms.ToPILImage()(avg_out*image).convert('RGB')
    avg_img.save('/data/pycode/FundusDR/imgs/IDRiD_085_avg.jpg')
    max_out, _ = torch.max(image, dim=0, keepdim=True)
    max_img = transforms.ToPILImage()(max_out*image).convert('RGB')
    max_img.save('/data/pycode/FundusDR/imgs/IDRiD_085_max.jpg')
    min_out, _ = torch.min(image, dim=0, keepdim=True)
    min_img = transforms.ToPILImage()(min_out*image).convert('RGB')
    min_img.save('/data/pycode/FundusDR/imgs/IDRiD_085_min.jpg')
    """
    """
    #https://zhuanlan.zhihu.com/p/74053773
    img_path = '/data/pycode/FundusDR/imgs/IDRiD_085.jpg'
    image = Image.open(img_path).convert('RGB')

    image_contour = image.filter(ImageFilter.SMOOTH)
    image_contour.save('/data/pycode/FundusDR/imgs/IDRiD_085_contour.jpg')
    """
    img_path = '/data/pycode/FundusDR/imgs/IDRiD_53.jpg'
    img = Image.open(img_path)
    img_horizontal = img.transpose(Image.FLIP_LEFT_RIGHT) #Flip left and right
    img_horizontal.save('/data/pycode/FundusDR/imgs/IDRiD_53_horizontal.jpg')
    img_vertical = img.transpose(Image.FLIP_TOP_BOTTOM) #Flip top and buttom
    img_vertical.save('/data/pycode/FundusDR/imgs/IDRiD_53_vertical.jpg')

def drawKeyPts(im,keyp,col,th):
    for curKey in keyp:
        x=np.int(curKey.pt[0])
        y=np.int(curKey.pt[1])
        size = np.int(curKey.size)
        cv2.circle(im, (x,y), size, col,thickness=th, lineType=8, shift=0) 
    plt.imshow(im)    
    return im    

def draw_matches(img1, kp1, img2, kp2, matches, color=None): 
    """Draws lines between matching keypoints of two images.  
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles 
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same 
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to 
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.  
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.  
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 15
    thickness = 5
    if color:
        c = color
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color: 
            c = np.random.randint(0,255,3) if len(img1.shape) == 3 else np.random.randint(0,256)
            c = (int(c[0]), int(c[1]), int(c[2]))
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)
    
    #plt.figure(figsize=(15,15))
    plt.imshow(new_img)
    return new_img

def getkeypoints():
    img_path = '/data/pycode/FundusDR/imgs/IDRiD_53.jpg'
    img = Image.open(img_path).convert('RGB')
    img_horizontal = img.transpose(Image.FLIP_LEFT_RIGHT)
    img_vertical = img.transpose(Image.FLIP_TOP_BOTTOM)
    orb = cv2.ORB_create()# Initiate SIFT detector
    img = np.array(img)
    img_horizontal = np.array(img_horizontal) 
    img_vertical = np.array(img_vertical) 
    kp, des = orb.detectAndCompute(img,None)
    kp_h, des_h = orb.detectAndCompute(img_horizontal,None)
    kp_v, des_v = orb.detectAndCompute(img_vertical,None)
    #draw
    #img = drawKeyPts(img.copy(),kp1,(0,255,0),5)
    #img = Image.fromarray(img).convert('RGB')
    #img.save('/data/pycode/FundusDR/imgs/IDRiD_53_kp.jpg')

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_h = bf.match(des,des_v)
    matches_h = sorted(matches_h, key = lambda x:x.distance)
    matches_v = bf.match(des,des_v)
    matches_v = sorted(matches_v, key = lambda x:x.distance)
    #Draw first 10 matches.
    #img = cv2.drawMatches(img,kp,img_horizontal,kp_h, matches_h[:20], None, flags=2)
    img = draw_matches(img,kp,img_horizontal,kp_h, matches_h[:20])
    img = Image.fromarray(img).convert('RGB')
    img.save('/data/pycode/FundusDR/imgs/IDRiD_53_match_h.jpg')

    
    """
    #knn match
    bf = cv2.BFMatcher()
    matches_h = bf.knnMatch(des,des_h, k=2) #bf.match(des,des_h)
    # Apply ratio test
    good = []
    for m,n in matches_h:
        if m.distance < 0.75*n.distance:
            good.append([m])
    img = cv2.drawMatchesKnn(img,kp,img_horizontal,kp_h, good, None, flags=2)
    img = Image.fromarray(img).convert('RGB')
    img.save('/data/pycode/FundusDR/imgs/IDRiD_53_match_h.jpg')
    """

def edgeDection():
    img_path = '/data/pycode/FundusDR/imgs/IDRiD_53.jpg'
    img = Image.open(img_path).convert('RGB')
    canny = cv2.Sobel(np.array(img), cv2.CV_64F, 1, 0, ksize=-1)
    img = Image.fromarray(canny).convert('RGB')
    img.save('/data/pycode/FundusDR/imgs/IDRiD_53_canny.jpg')

def transparent_back(img, cls='he'):
    #img = img.convert('RGBA')
    L, H = img.size
    color_0 = img.getpixel((0,0)) #alpha channel: 0~255
    for h in range(H):
        for l in range(L):
            dot = (l,h)
            color_1 = img.getpixel(dot)
            if color_1 == color_0:
                color_1 = color_1[:-1] + (0,)
                img.putpixel(dot,color_1)
            else: 
                if cls=='ma':
                    color_1 = ( 0, 0, 255, 255) #turn to blue  and transparency 
                    img.putpixel(dot,color_1)
                else: #'he'
                    color_1 = ( 0 , 255, 0, 255) #turn to green  and transparency 
                    img.putpixel(dot,color_1)
    return img

def RelevanceDegreeLoss():
    #query: TestingSet/IDRiD_56.jpg
    image = '/data/fjsdata/fundus/IDRID/ASegmentation/Images/TrainingSet/IDRiD_32.jpg' 
    mask = '/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TrainingSet/Haemorrhages/IDRiD_32_HE.tif'  
    #show 
    image = Image.open(image).convert('RGBA')
    mask = Image.open(mask).convert('RGBA')
    mask = transparent_back(mask, 'he')
    overlay = Image.alpha_composite(image, mask)
    plt.imshow(overlay)#cmap='gray'
    plt.axis('off')
    plt.savefig('/data/pycode/Thesis/imgs/IDRiD_32.jpg')

def main():
    #genMirrorImages()
    #getkeypoints()
    #edgeDection()
    RelevanceDegreeLoss()

if __name__ == '__main__':
    main()