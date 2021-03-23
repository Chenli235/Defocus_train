# -*- coding: utf-8 -*-

import os
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms,utils,models
import torchvision.utils as vutils
from torchvision import utils,models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#from pytorchcv.model_provider import get_model as ptcv_get_model
from scipy.interpolate import CubicSpline
import numpy as np
import timeit
import time
import random
from random import shuffle
import logging
import scipy.misc
import scipy.stats
import cv2
import skimage.io
import skimage.transform
import skimage.color
import skimage

from model.dataloader import Defocus_dataset_3,Normalizer, Augmenter
from model.data_test import test, test_perclass,test_wholeimage
logging.getLogger().setLevel(logging.DEBUG)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from model.dataloader import Defocus_dataset_3 ,Normalizer, Augmenter
from model.model import Unet, UNet, NestedUNet,Unet_3plus,Unet3,selfmade_net

image_size = 128

def sum(a,b):
    a = int(a)
    b = int(b)
    return a+b

def normalize_img(img):
    img = img.astype(np.float32)
    img[img>10000] = 10000
    #img[img<140] = 140
    mean = np.array([[[0]]])
    std = np.array([[10000]])
    #img = np.expand_dims(img,axis=2)
    img = (img-mean)/std
    return img

def img_to_batches(img):
    height = img.shape[2]
    width = img.shape[3]
    num = (height/image_size)*(width/image_size)
    image_batches = np.zeros((int(num),2,image_size,image_size))
    height_num = int(height/image_size)
    width_num = int(width/image_size)
    
    for i in range(height_num):
        for j in range(width_num):
            image_batches[i*width_num+j,:,:,:] = img[0,:,i*image_size:i*image_size+image_size,j*image_size:j*image_size+image_size]
    return image_batches

def cal_certainty(prob):
    sum_prob = np.sum(prob)
    num_classes = prob.shape[0]
    
    if sum_prob > 0:
        normalized_prob = prob/sum_prob
        certain_proxy = 1.0 - scipy.stats.entropy(normalized_prob)/np.log(num_classes)
    else:
        certain_proxy = 0.0
    certain_proxy = np.clip(certain_proxy,0.0,1.0)
    #print(certain_proxy)
    return certain_proxy

def get_certainty(prob):
    num_batches = prob.shape[0]
    #num_classes = prob.shape[1]
    cert = np.zeros(num_batches)
    for i in range(num_batches):
        cert[i] = cal_certainty(prob[i])
        
    return cert
    
def pred_whole(img):
    # choose network
    classifier = selfmade_net()
    #fc_features = classifier.fc.in_features
    #classifier.fc = nn.Linear(fc_features,11)
    model_name = 'selfmade_net_6nm_20000.pt'
    
    classifier = classifier.to(device)
    classifier.load_state_dict(torch.load(model_name))
    
    img = np.array(img)
    #img_small = img[1024-image_size*5:1024+image_size*5,1024-image_size*5:1024+image_size*5,:]
    img_small = img[128*4:128*12,128*4:128*12,:]
    img_norm = normalize_img(img_small) #2048*2048*3
    img_ = np.expand_dims(img_norm,axis = 0)
    img_whole = torch.from_numpy(img_.copy()).type(torch.FloatTensor).permute(0,3,1,2)
    #print(img_whole.shape)
    img_batches = img_to_batches(img_whole)
    classifier.eval()
    with torch.no_grad():
        output = classifier(torch.from_numpy(img_batches.copy()).type(torch.FloatTensor).to(device))
    
    pred = output.argmax(dim=1,keepdim=True).cpu().detach().numpy()
    prob = F.softmax(output,dim=1).cpu().detach().numpy()
    cert = get_certainty(prob)
    
    return pred,cert

def pred_onepatch(img):
    classifier = selfmade_net()
    model_name = 'selfmade_net_6nm_20000.pt'
    classifier = classifier.to(device)
    classifier.load_state_dict(torch.load(model_name))
    img = np.array(img)
    img_norm = normalize_img(img)
    img_ = np.expand_dims(img_norm,axis = 0)
    img__ = torch.from_numpy(img_.copy()).type(torch.FloatTensor).permute(0,3,1,2)
    classifier.eval()
    with torch.no_grad():
        output = classifier(img__.to(device))
    pred = output.argmax(dim=1,keepdim=True).cpu().detach().numpy()
    prob = F.softmax(output,dim=1).cpu().detach().numpy()
    cert = get_certainty(prob)
    return pred,cert