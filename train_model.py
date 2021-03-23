# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 19:01:25 2021

@author: ische
"""

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

from model.dataloader import Defocus_dataset_3, Defocus_dataset_2, Normalizer, Augmenter,Defocus_dataset_1
from model.data_test import test, test_perclass,test_wholeimage
logging.getLogger().setLevel(logging.DEBUG)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from model.dataloader import Defocus_dataset_3 ,Normalizer, Augmenter
from model.model import Unet, UNet, NestedUNet,Unet_3plus,Unet3,selfmade_net

if __name__ == "__main__":
    train_network = True
    if train_network == True:
        train_dir = 'train'
        test_dir = 'test'
        val_dir = 'val'
        logging.info('New train: loading data')
        train_dataset = Defocus_dataset_2(train_dir,True,transforms.Compose([Normalizer(),Augmenter()]))
        test_dataset = Defocus_dataset_2(test_dir,True,transforms.Compose([Normalizer(),Augmenter()]))
        val_dataset = Defocus_dataset_2(val_dir,True,transforms.Compose([Normalizer(),Augmenter()]))
        
        train_dataloader = DataLoader(train_dataset,batch_size = 8,shuffle=True,num_workers =4,pin_memory=True)
        test_dataloader = DataLoader(test_dataset,batch_size = 42,shuffle=True,num_workers =4,pin_memory=True)
        val_dataloader = DataLoader(val_dataset,batch_size = 42,shuffle=True,num_workers = 4,pin_memory=True)
        
        classifier = selfmade_net()
        classifier = classifier.to(device)
        optimizerC = optim.Adam(classifier.parameters(),lr=0.000005,betas=(0.5,0.999))
        criterion = nn.CrossEntropyLoss()
        
        epochs = 20000
        test_losses = []
        test_accuracies = []
        train_losses = []
        train_accuracies = []
        best_acc = 0.0
        
        for epoch in range(1,epochs+1):
            training_loss = 0.0
            running_loss = 0.0
            if epoch % 10 == 0:
                logging.info('\n Training Epoch: {}\n'.format(epoch))
            classifier.train()
            optimizerC.zero_grad()
            for index,data in enumerate(train_dataloader):
                optimizerC.zero_grad()
                if torch.cuda.is_available():
                    output = classifier(data['image'].to(device))
                    target = data['label'].type(dtype=torch.long).to(device)
                    loss = criterion(output,target)
                    loss.backward()
                    optimizerC.step()
                    running_loss += loss.item()
                    training_loss += loss.item()
                    if index % 30 == 29:
                        logging.info('\n Epoch: {}/{}  Loss: {}\n'.format(epoch,index+1,running_loss/30))
                        running_loss = 0
            train_losses.append(training_loss)
            
            
            # save the model every few epochs
            if epoch % 1000 == 0:
                logging.info('Epoch: {}'.format(epoch))
                torch.save(classifier.state_dict(),'saved_model/selfmade_net_{}nm_{}.pt'.format(train_dataset.distance,epoch))
                test_loss, test_accuracy = test(classifier,val_dataloader,len(val_dataset),True)
                test_losses.append(test_loss)
                test_accuracies.append(test_accuracy)
            else:
                test_loss, test_accuracy = test(classifier,val_dataloader,len(val_dataset),False)
                test_losses.append(test_loss)
                test_accuracies.append(test_accuracy)
            np.savetxt('selfmade_test_losses.txt',test_losses,delimiter=',')
            np.savetxt('selfmade_test_accuracies.txt',test_accuracies,delimiter=',')
            np.savetxt('selfmade_train_losses.txt',train_losses,delimiter=',')