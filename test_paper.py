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

from model.dataloader import Defocus_dataset_3,Normalizer, Augmenter
from model.data_test import test, test_perclass,test_wholeimage

logging.getLogger().setLevel(logging.DEBUG)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from model.dataloader import Defocus_dataset_3 ,Defocus_dataset_2, Normalizer, Augmenter,Defocus_dataset_1
from model.model import Unet, UNet, NestedUNet,Unet_3plus,Unet3,selfmade_net
image_size = 128
#model_input = 1


if __name__ == "__main__":
    test_all.test_differentzstack()
    #test_all.test_differentzspacing()
    # file = open(file_validloss_1stack)
    # value_1 = []
    # while 1:
    #     lines = file.readlines(30000)
    #     if not lines:
    #         break
    #     for line in lines:
    #         value_1.append(float(line))
    #         pass
    # file.close()
    
    # file = open(file_validloss_2stack)
    # value_2 = []
    # while 1:
    #     lines = file.readlines(30000)
    #     if not lines:
    #         break
    #     for line in lines:
    #         value_2.append(float(line))
    #         pass
    # file.close()
    
    # file = open(file_validloss_3stack)
    # value_3 = []
    # while 1:
    #     lines = file.readlines(30000)
    #     if not lines:
    #         break
    #     for line in lines:
    #         value_3.append(float(line))
    #         pass
    # file.close()
    
    # smoothed_1 = []
    # last = value_1[0]
    # weight = 0.95
    # for i in value_1:
    #     smoothed_val = last*weight + (1-weight)*i
    #     smoothed_1.append(smoothed_val)
    #     last = smoothed_val
        
    # smoothed_2 = []
    # last = value_2[0]
    # weight = 0.95
    # for i in value_2:
    #     smoothed_val = last*weight + (1-weight)*i
    #     smoothed_2.append(smoothed_val)
    #     last = smoothed_val
        
    # smoothed_3 = []
    # last = value_3[0]
    # weight = 0.95
    # for i in value_3:
    #     smoothed_val = last*weight + (1-weight)*i
    #     smoothed_3.append(smoothed_val)
    #     last = smoothed_val
    
    # plt.figure(1)
    # plt.plot(range(20000),np.array(smoothed_1),range(20000),np.array(smoothed_2),range(20000),np.array(smoothed_3))
    # plt.legend(['z-stack-1','z-stack-2','z-stack-3'])
    # plt.xlabel('Epochs')
    # plt.ylabel('Validation Loss')
    
    
    model_1stack = 'differstack_sameoutput/1stack/selfmade_net_6nm_20000.pt'
    model_2stack = 'differstack_sameoutput/2stack/selfmade_net_6nm_20000.pt'
    model_3stack = 'differstack_sameoutput/3stack/selfmade_net_6nm_20000.pt'
    
    
    