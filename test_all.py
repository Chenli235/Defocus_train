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

import skimage.io
import skimage.transform
import skimage.color
import skimage
import test_all

from model.dataloader import Defocus_dataset_3,Defocus_dataset_2,Defocus_dataset_1,Normalizer, Augmenter
from model.data_test import test, test_perclass,test_wholeimage

logging.getLogger().setLevel(logging.DEBUG)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from model.dataloader import Defocus_dataset_3 ,Defocus_dataset_2, Normalizer, Augmenter,Defocus_dataset_1
from model.model import Unet, UNet, NestedUNet,Unet_3plus,Unet3,selfmade_net
image_size = 128

def test_differentzstack():
    file_trainingloss_1stack = 'differstack_sameoutput/1stack/selfmade_train_losses.txt'
    file_trainingloss_2stack = 'differstack_sameoutput/2stack/selfmade_train_losses.txt'
    file_trainingloss_3stack = 'differstack_sameoutput/3stack/selfmade_train_losses.txt'
    
    file_validaccuracy_1stack = 'differstack_sameoutput/1stack/selfmade_test_accuracies.txt'
    file_validaccuracy_2stack = 'differstack_sameoutput/2stack/selfmade_test_accuracies.txt'
    file_validaccuracy_3stack = 'differstack_sameoutput/3stack/selfmade_test_accuracies.txt'
    
    file = open(file_trainingloss_1stack)
    value_1 = []
    while 1:
        lines = file.readlines(30000)
        if not lines:
            break
        for line in lines:
            value_1.append(float(line))
            pass
    file.close()
    
    file = open(file_trainingloss_2stack)
    value_2 = []
    while 1:
        lines = file.readlines(30000)
        if not lines:
            break
        for line in lines:
            value_2.append(float(line))
            pass
    file.close()
    
    file = open(file_trainingloss_3stack)
    value_3 = []
    while 1:
        lines = file.readlines(30000)
        if not lines:
            break
        for line in lines:
            value_3.append(float(line))
            pass
    file.close()
    
    smoothed_1 = []
    last = value_1[0]
    weight = 0.95
    for i in value_1:
        smoothed_val = last*weight + (1-weight)*i
        smoothed_1.append(smoothed_val)
        last = smoothed_val
        
    smoothed_2 = []
    last = value_2[0]
    weight = 0.95
    for i in value_2:
        smoothed_val = last*weight + (1-weight)*i
        smoothed_2.append(smoothed_val)
        last = smoothed_val
        
    smoothed_3 = []
    last = value_3[0]
    weight = 0.95
    for i in value_3:
        smoothed_val = last*weight + (1-weight)*i
        smoothed_3.append(smoothed_val)
        last = smoothed_val
    
    plt.figure(1)
    plt.plot(range(20000),np.array(smoothed_1),range(20000),np.array(smoothed_2),range(20000),np.array(smoothed_3))
    plt.legend(['1 defocused image','2 defocused images','3 defocused images'],fontsize=18)
    #plt.xlabel('Epochs')
    #plt.ylabel('Train Loss')
    
    
    file = open(file_validaccuracy_1stack)
    value_1 = []
    while 1:
        lines = file.readlines(30000)
        if not lines:
            break
        for line in lines:
            value_1.append(float(line))
            pass
    file.close()
    
    file = open(file_validaccuracy_2stack)
    value_2 = []
    while 1:
        lines = file.readlines(30000)
        if not lines:
            break
        for line in lines:
            value_2.append(float(line))
            pass
    file.close()
    
    file = open(file_validaccuracy_3stack)
    value_3 = []
    while 1:
        lines = file.readlines(30000)
        if not lines:
            break
        for line in lines:
            value_3.append(float(line))
            pass
    file.close()
    
    smoothed_1 = []
    last = value_1[0]
    weight = 0.95
    for i in value_1:
        smoothed_val = last*weight + (1-weight)*i
        smoothed_1.append(smoothed_val)
        last = smoothed_val
        
    smoothed_2 = []
    last = value_2[0]
    weight = 0.95
    for i in value_2:
        smoothed_val = last*weight + (1-weight)*i
        smoothed_2.append(smoothed_val)
        last = smoothed_val
        
    smoothed_3 = []
    last = value_3[0]
    weight = 0.95
    for i in value_3:
        smoothed_val = last*weight + (1-weight)*i
        smoothed_3.append(smoothed_val)
        last = smoothed_val
    
    plt.figure(2)
    plt.plot(range(20000),np.array(smoothed_1),range(20000),np.array(smoothed_2),range(20000),np.array(smoothed_3))
    plt.legend(['1 defocused image','2 defocused images','3 defocused images'],fontsize=18)
    #plt.xlabel('Epochs')
    #plt.ylabel('classificaton rate on validation dataset(%)')
    
    file_trainingloss_2um = 'differentzspacing_sameoutput/2um/selfmade_test_losses.txt'
    file_trainingloss_4um = 'differentzspacing_sameoutput/4um/selfmade_test_losses.txt'
    file_trainingloss_6um = 'differentzspacing_sameoutput/6um/selfmade_test_losses.txt'
    file_trainingloss_8um = 'differentzspacing_sameoutput/8um/selfmade_test_losses.txt'
    file_trainingloss_10um = 'differentzspacing_sameoutput/10um/selfmade_test_losses.txt'
    
    file_validaccuracy_2um = 'differentzspacing_sameoutput/2um/selfmade_test_accuracies.txt'
    file_validaccuracy_4um = 'differentzspacing_sameoutput/4um/selfmade_test_accuracies.txt'
    file_validaccuracy_6um = 'differentzspacing_sameoutput/6um/selfmade_test_accuracies.txt'
    file_validaccuracy_8um = 'differentzspacing_sameoutput/8um/selfmade_test_accuracies.txt'
    file_validaccuracy_10um = 'differentzspacing_sameoutput/10um/selfmade_test_accuracies.txt'
    
    
    file = open(file_trainingloss_2um)
    value_1 = []
    while 1:
        lines = file.readlines(30000)
        if not lines:
            break
        for line in lines:
            value_1.append(float(line))
            pass
    file.close()
    
    file = open(file_trainingloss_4um)
    value_2 = []
    while 1:
        lines = file.readlines(30000)
        if not lines:
            break
        for line in lines:
            value_2.append(float(line))
            pass
    file.close()
    
    file = open(file_trainingloss_6um)
    value_3 = []
    while 1:
        lines = file.readlines(30000)
        if not lines:
            break
        for line in lines:
            value_3.append(float(line))
            pass
    file.close()
    
    file = open(file_trainingloss_8um)
    value_4 = []
    while 1:
        lines = file.readlines(30000)
        if not lines:
            break
        for line in lines:
            value_4.append(float(line))
            pass
    file.close()
    
    file = open(file_trainingloss_10um)
    value_5 = []
    while 1:
        lines = file.readlines(30000)
        if not lines:
            break
        for line in lines:
            value_5.append(float(line))
            pass
    file.close()
    
    smoothed_1 = []
    last = value_1[0]
    weight = 0.98
    for i in value_1:
        smoothed_val = last*weight + (1-weight)*i
        smoothed_1.append(smoothed_val)
        last = smoothed_val
        
    smoothed_2 = []
    last = value_2[0]
    weight = 0.98
    for i in value_2:
        smoothed_val = last*weight + (1-weight)*i
        smoothed_2.append(smoothed_val)
        last = smoothed_val
        
    smoothed_3 = []
    last = value_3[0]
    weight = 0.98
    for i in value_3:
        smoothed_val = last*weight + (1-weight)*i
        smoothed_3.append(smoothed_val)
        last = smoothed_val
        
    smoothed_4 = []
    last = value_4[0]
    weight = 0.98
    for i in value_4:
        smoothed_val = last*weight + (1-weight)*i
        smoothed_4.append(smoothed_val)
        last = smoothed_val
        
    smoothed_5 = []
    last = value_5[0]
    weight = 0.98
    for i in value_5:
        smoothed_val = last*weight + (1-weight)*i
        smoothed_5.append(smoothed_val)
        last = smoothed_val
    
    plt.figure(3)
    plt.plot(range(20000),np.array(smoothed_1),range(20000),np.array(smoothed_2),range(20000),np.array(smoothed_3),range(20000),np.array(smoothed_4),range(20000),np.array(smoothed_5))
    plt.legend(['2um','4um','6um','8um','10um'],fontsize=18)
    #plt.xlabel('Epochs')
    #plt.ylabel('Train Loss')
    
    file = open(file_validaccuracy_2um)
    value_1 = []
    while 1:
        lines = file.readlines(30000)
        if not lines:
            break
        for line in lines:
            value_1.append(float(line))
            pass
    file.close()
    
    file = open(file_validaccuracy_4um)
    value_2 = []
    while 1:
        lines = file.readlines(30000)
        if not lines:
            break
        for line in lines:
            value_2.append(float(line))
            pass
    file.close()
    
    file = open(file_validaccuracy_6um)
    value_3 = []
    while 1:
        lines = file.readlines(30000)
        if not lines:
            break
        for line in lines:
            value_3.append(float(line))
            pass
    file.close()
    
    file = open(file_validaccuracy_8um)
    value_4 = []
    while 1:
        lines = file.readlines(30000)
        if not lines:
            break
        for line in lines:
            value_4.append(float(line))
            pass
    file.close()
    
    file = open(file_validaccuracy_10um)
    value_5 = []
    while 1:
        lines = file.readlines(30000)
        if not lines:
            break
        for line in lines:
            value_5.append(float(line))
            pass
    file.close()
    
    smoothed_1 = []
    last = value_1[0]
    weight = 0.98
    for i in value_1:
        smoothed_val = last*weight + (1-weight)*i
        smoothed_1.append(smoothed_val)
        last = smoothed_val
        
    smoothed_2 = []
    last = value_2[0]
    weight = 0.98
    for i in value_2:
        smoothed_val = last*weight + (1-weight)*i
        smoothed_2.append(smoothed_val)
        last = smoothed_val
        
    smoothed_3 = []
    last = value_3[0]
    weight = 0.98
    for i in value_3:
        smoothed_val = last*weight + (1-weight)*i
        smoothed_3.append(smoothed_val)
        last = smoothed_val
        
    smoothed_4 = []
    last = value_4[0]
    weight = 0.98
    for i in value_4:
        smoothed_val = last*weight + (1-weight)*i
        smoothed_4.append(smoothed_val)
        last = smoothed_val
        
    smoothed_5 = []
    last = value_5[0]
    weight = 0.98
    for i in value_5:
        smoothed_val = last*weight + (1-weight)*i
        smoothed_5.append(smoothed_val)
        last = smoothed_val
    
    plt.figure(4)
    plt.plot(range(20000),np.array(smoothed_1),range(20000),np.array(smoothed_2),range(20000),np.array(smoothed_3),range(20000),np.array(smoothed_4),range(20000),np.array(smoothed_5))
    plt.legend(['2um','4um','6um','8um','10um'],fontsize=18)
    #plt.xlabel('Epochs')
    #plt.ylabel('classificaton rate on validation dataset(%)')
    
    
def test_differentzspacing():
    model_1stack = 'differstack_sameoutput/1stack/selfmade_net_6nm_20000.pt'
    model_2stack = 'differstack_sameoutput/2stack/selfmade_net_6nm_20000.pt'
    model_3stack = 'differstack_sameoutput/3stack/selfmade_net_6nm_20000.pt'
    test_dir = 'F:/lightsheet_data/test'
    # for 1-stack
    # test_dataset = Defocus_dataset_1(test_dir,True,transforms.Compose([Normalizer(),Augmenter()]))
    # test_dataloader = DataLoader(test_dataset,batch_size = 1,shuffle=True,num_workers =1,pin_memory=False)
    # classifier_1 = selfmade_net()
    # classifier_1.load_state_dict(torch.load(model_1stack))
    # classifier_1.to(device)
    # test_perclass(classifier_1, test_dataloader, 0,display=False,total_round=10*4)
    
    # for 1-stack
    # test_dataset = Defocus_dataset_2(test_dir,True,transforms.Compose([Normalizer(),Augmenter()]))
    # test_dataloader = DataLoader(test_dataset,batch_size = 1,shuffle=True,num_workers =1,pin_memory=False)
    # classifier_2 = selfmade_net()
    # classifier_2.load_state_dict(torch.load(model_2stack))
    # classifier_2.to(device)
    # test_perclass(classifier_2, test_dataloader, 0,display=False,total_round=10*4)
    # for 3-stack
    test_dataset = Defocus_dataset_3(test_dir,True,transforms.Compose([Normalizer(),Augmenter()]))
    test_dataloader = DataLoader(test_dataset,batch_size = 1,shuffle=True,num_workers =1,pin_memory=False)
    classifier_3 = selfmade_net()
    classifier_3.load_state_dict(torch.load(model_3stack))
    classifier_3.to(device)
    test_perclass(classifier_3, test_dataloader, 0,display=False,total_round=10*4)