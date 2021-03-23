# -*- coding: utf-8 -*-

import sys
import os
import torch
import numpy as np
import random
import csv
import os
import random
from random import shuffle
from os import listdir
from os.path import join
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils,models
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import choice
import logging
import model
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.misc
import scipy.stats
import cv2


import skimage.io
import skimage.transform
import skimage.color
import skimage
from sklearn.metrics import confusion_matrix

from PIL import Image
import seaborn as sns

logging.getLogger().setLevel(logging.INFO)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
image_size = 128

def test(model,test_loader,m_test,display=False,total_round=10):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    for i in range(total_round):
        with torch.no_grad():
            for batch in test_loader:
                output = model(batch['image'].to(device))
                target = batch['label'].type(dtype=torch.long).to(device)
                test_loss += float(criterion(output,target).item())
                pred = output.argmax(dim=1,keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= (m_test*total_round)
    if display == True:
        logging.info('\ndataset: Average loss: {:.4f}, Accuracy:{}/{}  ({:.2f}%)\n'.format(test_loss,correct,m_test*total_round,100.*correct/m_test/total_round))
    return test_loss, 100*correct/m_test/total_round

def test_perclass(model,test_loader,m_test,display=False,total_round = 10):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct_perclass = np.zeros(13)
    sum_perclass = np.zeros(13)
    target_point = []
    pred_point = []
    for i in range(total_round):
        print(i)
        with torch.no_grad():
            for batch in test_loader:
                output = model(batch['image'].to(device))
                target = batch['label'].type(dtype=torch.long).to(device)
                sum_perclass[target] += 1
                prob = F.softmax(output,dim=1).cpu().detach().numpy()
                certainty = get_certainty(prob)
                pred = output.argmax(dim=1,keepdim=True)
                target_point.append(target.item())
                pred_point.append(pred.item())
                if pred.eq(target.view_as(pred)).sum().item() == 1 and certainty > 0.3:
                    correct_perclass[target] += 1
    name_list = ['-40um','-32um','-24um','-16um','-8um','0um','8um','16um','24um','32um','40um']
    C2 = confusion_matrix(target_point,pred_point)
    # plt.figure(1)
    # plt.bar(range(11),correct_perclass/sum_perclass,tick_label=name_list)
    # plt.ylabel('accuracy')
    
    plt.figure(2)
    sns.heatmap(C2,annot=False,vmax=100)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    plt.xticks([])
    plt.yticks([])
    return correct_perclass ,sum_perclass,correct_perclass/sum_perclass

def draw_caption(img,pred,prob,cert,draw_label):
    labels={}
    labels[0] = '-40um';labels[1] = '-32um';labels[2] = '-24um'; labels[3] = '-16um';labels[4] = '-8um';
    labels[5] = '0um'; labels[6] = '8um'; labels[7] = '16um'; labels[8] = '24um';labels[9] = '32um'; labels[10] = '40um'
    #print(cert)
    #print(img.shape)
    min = 0
    max = 1
    img = img[1]
    if cert>0.4:
        0
        #img = (img-min)/(max-min)*1
    #print(np.max(img))
    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)
    if draw_label == True:
        if cert>0.3:
            #print(img.shape)
            0
            #img = cv2.putText(img, labels[int(pred)], (2,20), cv2.FONT_HERSHEY_PLAIN, 1.5, (1,0,0),1)
    #print(img.shape)
    #print(pred)
    return img

def normalize_img(img):
    img = img.astype(np.float32)
    img[img>10000] = 10000
    #img[img<140] = 140
    mean = np.array([[[0]]])
    std = np.array([[10000]])
    #img = np.expand_dims(img,axis=2)
    img = (img-mean)/std
    return img

def cal_certainty(prob):
    sum_prob = np.sum(prob)
    num_classes = prob.shape[0]
    if sum_prob > 0:
        normalized_prob = prob/sum_prob
        certain_proxy = 1.0 - scipy.stats.entropy(normalized_prob)/np.log(num_classes)
    else:
        certain_proxy = 0.0
    certain_proxy = np.clip(certain_proxy,0.0,1.0)
    return certain_proxy
def get_certainty(prob):
    num_batches = prob.shape[0]
    num_classes = prob.shape[1]
    cert = np.zeros(num_batches)
    for i in range(num_batches):
        cert[i] = cal_certainty(prob[i])
    return cert

def image_allbatches(img):
    height = img.shape[2]
    width = img.shape[3]
    height_num = int(height/image_size)
    width_num = int(width/image_size)
    image_batches = np.zeros((height_num*width_num,3,image_size,image_size))

    for i in range(0,height_num):
        for j in range(0,width_num):
            image_batches[i*width_num+j,:,:,:] = img[:,:,32+i*124:32+i*124+124,32+j*124:32+j*124+124]
    return image_batches

def get_calss_rgb(pred):
    227,29,22
    227,132,22
    227,216,22
    132,227,22
    22,227,106
    22,227,209
    22,151,227
    22,61,227
    91,22,227
    168,22,227
    227,22,156
    if pred == 0:
        return (227/255,29/255,22/255)
    elif pred == 1:
        return (227/255,132/255,22/255)
    elif pred == 2:
        return (227/255,216/255,22/255)
    elif pred == 3:
        return (132/255,227/255,22/255)
    elif pred == 4:
        return (22/255,227/255,106/255)
    elif pred == 5:
        return (22/255,227/255,209/255)
    elif pred == 6:
        return (22/255,151/255,227/255)
    elif pred == 7:
        return (22/255,61/255,227/255)
    elif pred == 8:
        return (91/255,22/255,227/255)
    elif pred == 9:
        return (168/255,22/255,227/255)
    elif pred == 10:
        return (227/255,22/255,156/255)
    
    # if pred == 0 or pred == 10:
    #     return (1,0,0) # red
    # elif pred == 1 or pred == 9:
    #     return (1,1,0) # yellow
    # elif pred == 2 or pred == 8:
    #     return (0,1,0) # green
    # elif pred == 3 or pred == 7:
    #     return (0,1,1) # cyan
    # elif pred == 4 or pred == 6:
    #     return (0,0,1) # blue
    # elif pred == 5:
    #     return (1,0,1) # magenta

def set_border_pixel(img,color_border):
    border_size = 4
    for i in range(3):
        img[:,0:border_size,i] = color_border[i]
        img[:,img.shape[1]-border_size:img.shape[1],i] = color_border[i]
        img[0:border_size,:,i] = color_border[i]
        img[img.shape[i]-border_size:img.shape[i],:,i] = color_border[i]
    return img
    
def add_border_img(labeled_img,pred,prob,certainty,draw_label):
    for i in range(labeled_img.shape[0]):
        img = labeled_img[i]
        color_border = get_calss_rgb(int(pred[i]))
        if certainty[i]>0.3:
            img = set_border_pixel(img,color_border)
        labeled_img[i] = img
    return labeled_img
    
def imgbatches_toimg(img_batches):
    height_num = 16
    width_num = 16
    whole_img = np.zeros((height_num*image_size,width_num*image_size,3))
    for i in range(height_num):
        for j in range(width_num):
            whole_img[i*image_size:i*image_size+image_size,j*image_size:j*image_size+image_size,:] = img_batches[i*width_num+j]
            
    return whole_img

def convertto_visable(img_batches,certainty):
    new_batches = np.zeros((16*16,image_size,image_size,3))
    for i in range(img_batches.shape[0]):
        if certainty[i]>0.3:
            new_batches[i] = img_batches[i]/np.max(img_batches[i])*0.8
        else:
            new_batches[i] = img_batches[i]
    return new_batches
def test_wholeimage(img,model,draw_label = False):
    model.eval()
    img_old = img[32:2016,32:2016,1]
    #print(img_old.shape)
    #plt.figure(3)
    #plt.imshow(img_old,vmax=2000,cmap='gray')
    img = normalize_img(img)
    img = np.expand_dims(img,axis = 0)
    img = torch.from_numpy(img.copy()).type(torch.FloatTensor).permute(0,3,1,2)
    img_batches = image_allbatches(img)
    #print(img_batches.shape)
    #img_batches_expandRGB = convertto_visable(img_batches)
    with torch.no_grad():
        output1 = model(torch.from_numpy(img_batches[:64].copy()).type(torch.FloatTensor).to(device))
        output2 = model(torch.from_numpy(img_batches[64:128].copy()).type(torch.FloatTensor).to(device))
        output3 = model(torch.from_numpy(img_batches[128:192].copy()).type(torch.FloatTensor).to(device))
        output4 = model(torch.from_numpy(img_batches[192:256].copy()).type(torch.FloatTensor).to(device))

    # output1 = model(torch.from_numpy(img_batches[:32].copy()).type(torch.FloatTensor).to(device))
    # output2 = model(torch.from_numpy(img_batches[32:64].copy()).type(torch.FloatTensor).to(device))
    # output3 = model(torch.from_numpy(img_batches[64:96].copy()).type(torch.FloatTensor).to(device))
    # output4 = model(torch.from_numpy(img_batches[96:128].copy()).type(torch.FloatTensor).to(device))
    # torch.cuda.empty_cache()
    # output5 = model(torch.from_numpy(img_batches[128:160].copy()).type(torch.FloatTensor).to(device))
    # output6 = model(torch.from_numpy(img_batches[160:192].copy()).type(torch.FloatTensor).to(device))
    # output7 = model(torch.from_numpy(img_batches[192:224].copy()).type(torch.FloatTensor).to(device))
    # output8 = model(torch.from_numpy(img_batches[224:256].copy()).type(torch.FloatTensor).to(device))
    # torch.cuda.empty_cache()
    
    output = torch.cat((output1,output2,output3,output4))
    
    pred = output.argmax(dim=1,keepdim=True).cpu().detach().numpy()
    prob = F.softmax(output,dim=1).cpu().detach().numpy()
    certainty = get_certainty(prob)
    labeled_img = np.zeros((img_batches.shape[0],image_size,image_size,3))

    for i  in range(img_batches.shape[0]):#
        img = draw_caption(img_batches[i],pred[i],prob[i],certainty[i],draw_label)
        #print(img.shape)
        #plt.imshow(img,vmin=0,vmax=2000)
        labeled_img[i] = img
    #print(pred)
    labeled_img_convert = convertto_visable(labeled_img,certainty)
    labeled_border_img = add_border_img(labeled_img_convert,pred,prob,certainty,draw_label)
    #print(labeled_border_img.shape)
    whole_labeled_img = imgbatches_toimg(labeled_border_img)
    #print(whole_labeled_img.shape)
    fig= plt.imshow(whole_labeled_img)
    
    
    plt.imsave('test.tiff', whole_labeled_img)
    # generate color bar
    #plt.figure(2)
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    cmap = mpl.colors.ListedColormap(['#e31d16','#e38416','#e3d816','#84e316','#16e36a','#16e3d1','#1697e3','#163de3','#5b16e3','#a816e3','#e3169c'])
    # RGB value from -40um to 40um

    cmap.set_over('0.25')
    cmap.set_under('0.75')
    bounds = [0,1, 2, 3, 4,5,6,7,8,9,10,11]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    fig.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
    cax=ax,
    boundaries=bounds,
    extend='neither',
    ticks=[],
    spacing='proportional',
    orientation='horizontal',
    #label='Discrete intervals, some other units',
)
    return 0

if __name__=="__main__":
    model_name = '../saved_model/model_Resnet34_2000.pt'
    classifier = models.resnet34()
    fc_features = classifier.fc.in_features
    classifier.fc = nn.Linear(fc_features,11)

    classifier = classifier.to(device)
    classifier.load_state_dict(torch.load(model_name))
    
    img0_name = 'E:/imagedata/10_20_2020_cochlea/561nm/182900_258200_467330_10.tiff'
    img1_name = 'E:/imagedata/10_20_2020_cochlea/561nm/182900_258200_467330_14.tiff'
    img2_name = 'E:/imagedata/10_20_2020_cochlea/561nm/182900_258200_467330_18.tiff'
    
    img0_name = 'E:/imagedata/10_20_2020_cochlea/561nm/180400_262200_468130_22.tiff'
    img1_name = 'E:/imagedata/10_20_2020_cochlea/561nm/180400_262200_468130_26.tiff'
    img2_name = 'E:/imagedata/10_20_2020_cochlea/561nm/180400_262200_468130_30.tiff'
    
    img0_name = 'E:/imagedata/10_20_2020_cochlea/561nm/175900_257200_464730_26.tiff'
    img1_name = 'E:/imagedata/10_20_2020_cochlea/561nm/175900_257200_464730_30.tiff'
    img2_name = 'E:/imagedata/10_20_2020_cochlea/561nm/175900_257200_464730_34.tiff'
    
    img0_name = 'E:/spericalabration_dataset/lightsheet_data/10_23_2020_cochlea/561nm/177300_129200_418760_42.tiff'
    img1_name = 'E:/spericalabration_dataset/lightsheet_data/10_23_2020_cochlea/561nm/177300_129200_418760_46.tiff'
    img2_name = 'E:/spericalabration_dataset/lightsheet_data/10_23_2020_cochlea/561nm/177300_129200_418760_50.tiff'
    
    img0_name = 'E:/spericalabration_dataset/lightsheet_data/10_23_2020_cochlea/561nm/179300_133200_422760_34.tiff'
    img1_name = 'E:/spericalabration_dataset/lightsheet_data/10_23_2020_cochlea/561nm/179300_133200_422760_38.tiff'
    img2_name = 'E:/spericalabration_dataset/lightsheet_data/10_23_2020_cochlea/561nm/179300_133200_422760_42.tiff'
    
    img0 = skimage.io.imread(img0_name)
    img1 = skimage.io.imread(img1_name)
    img2 = skimage.io.imread(img2_name)
    img = np.dstack((img0,img1,img2))
    test_wholeimage(img,classifier,True)