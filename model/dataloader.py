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
from random import choice
from os import listdir
from os.path import join
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
import matplotlib.pyplot as plt

import skimage.io
import skimage.transform
import skimage.color
import skimage

from PIL import Image
image_size = 128

def is_tiff(filename):
    return any(filename.endswith(extension) for extension in ['.tif','.tiff'])

class Defocus_dataset_3(Dataset):
    def __init__(self,image_dir,labeled,transform=None,random_shuffle=True):
        self.images_dir = image_dir
        self.infocuse_dir = image_dir + '/26'
        self.transform = transform
        self.labeled = labeled
        self.image_files = np.array([join(image_dir,file_name) for file_name in sorted(listdir(self.infocuse_dir)) if is_tiff(file_name)])
        if random_shuffle:
            shuffle(self.image_files)
        #self.levels = [12,14,16,18,20,22,24,26,28,30,32,34,36,38,40] # 4um
        #self.levels = [5,8,11,14,17,20,23,26,29,32,35,38,41,44,47] # 6um
        self.model_input = 2
        self.distance = 6
        if self.distance == 4:
            self.levels = [16,18,20,22,24,26,28,30,32,34,36] # 4um
        elif self.distance == 6:
            self.levels = [8,11,14,17,20,23,26,29,32,35,38,41,44] # 6um
        elif self.distance == 8:
            self.levels = [6,10,14,18,22,26,30,34,38,42,46] # 8um
        #self.levels = [26,31,36,41,46,51]
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self,index):
        image,image_name = self.read_image(index)
        if self.labeled:
            label = self.read_label(image_name)
        else:
            label = None
        sample = {'image':image,'label':label}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def read_image(self,index):
        match_word = self.image_files[index][-28:][:20]
        level_1 = choice(self.levels)
        
        if self.distance == 4:
            level_0 = level_1 - 2
            level_2 = level_1 + 2
        elif self.distance == 6:
            level_0 = level_1 - 3
            level_2 = level_1 + 3
        elif self.distance == 8:
            level_0 = level_1 - 3
            level_2 = level_1 + 3
            if self.model_input == 3:
                level_0 = level_1 - 4
                level_2 = level_1 + 4
            elif self.model_input == 2:
                level_0 = level_1 -4
                level_2 = level_1
            elif self.model_input == 1:
                level_0 = level_1
                level_2 = level_1
            
        image_path_1 = self.images_dir + '/' + str(level_1)
        image_path_0 = self.images_dir + '/' + str(level_0)
        image_path_2 = self.images_dir + '/' + str(level_2)
        img_name = image_path_1 + '/' + match_word + '_' +str(level_1).zfill(2) + '.tiff'
        
        image_1 = skimage.io.imread(image_path_1 + '/' + match_word + '_' +str(level_1).zfill(2) + '.tiff')
        image_0 = skimage.io.imread(image_path_0 + '/' + match_word + '_' +str(level_0).zfill(2) + '.tiff')
        image_2 = skimage.io.imread(image_path_2 + '/' + match_word + '_' +str(level_2).zfill(2) + '.tiff')
        
        
        image_1[image_1>10000] = 10000
        image_0[image_0>10000] = 10000
        image_2[image_2>10000] = 10000
        
        x = random.randint(0, image_1.shape[1]-image_size)
        y = random.randint(0,image_1.shape[0]-image_size)
        img_1 = image_1[y:y+image_size,x:x+image_size]
        img_0 = image_0[y:y+image_size,x:x+image_size]
        img_2 = image_2[y:y+image_size,x:x+image_size]
        img = np.dstack((img_0,img_1,img_2))
        
        
        if np.random.rand() > 0.8:
            scale = np.random.uniform(0.98,1.2)
            img = img*scale
        return img,img_name
    
    def read_label(self,filename):

        # 4um
        if self.distance == 4:
            if filename.endswith('_16.tiff'):
                return 0
            elif filename.endswith('_18.tiff'):
                return 1
            elif filename.endswith('_20.tiff'):
                return 2
            elif filename.endswith('_22.tiff'):
                return 3
            elif filename.endswith('_24.tiff'):
                return 4
            elif filename.endswith('_26.tiff'):
                return 5
            elif filename.endswith('_28.tiff'):
                return 6
            elif filename.endswith('_30.tiff'):
                return 7
            elif filename.endswith('_32.tiff'):
                return 8
            elif filename.endswith('_34.tiff'):
                return 9
            elif filename.endswith('_36.tiff'):
                return 10
        elif self.distance == 6:
            if filename.endswith('_08.tiff'):
                return 0
            elif filename.endswith('_11.tiff'):
                return 1
            elif filename.endswith('_14.tiff'):
                return 2
            elif filename.endswith('_17.tiff'):
                return 3
            elif filename.endswith('_20.tiff'):
                return 4
            elif filename.endswith('_23.tiff'):
                return 5
            elif filename.endswith('_26.tiff'):
                return 6
            elif filename.endswith('_29.tiff'):
                return 7
            elif filename.endswith('_32.tiff'):
                return 8
            elif filename.endswith('_35.tiff'):
                return 9
            elif filename.endswith('_38.tiff'):
                return 10
            elif filename.endswith('_41.tiff'):
                return 11
            elif filename.endswith('_44.tiff'):
                return 12
                
                
        elif self.distance == 8:
            if filename.endswith('_06.tiff'):
                return 0
            elif filename.endswith('_10.tiff'):
                return 1
            elif filename.endswith('_14.tiff'):
                return 2
            elif filename.endswith('_18.tiff'):
                return 3
            elif filename.endswith('_22.tiff'):
                return 4
            elif filename.endswith('_26.tiff'):
                return 5
            elif filename.endswith('_30.tiff'):
                return 6
            elif filename.endswith('_34.tiff'):
                return 7
            elif filename.endswith('_38.tiff'):
                return 8
            elif filename.endswith('_42.tiff'):
                return 9
            elif filename.endswith('_46.tiff'):
                return 10
        
class Defocus_dataset_2(Dataset):
    def __init__(self,image_dir,labeled,transform=None,random_shuffle=True):
        self.images_dir = image_dir
        self.infocuse_dir = image_dir + '/26'
        self.transform = transform
        self.labeled = labeled
        self.image_files = np.array([join(image_dir,file_name) for file_name in sorted(listdir(self.infocuse_dir)) if is_tiff(file_name)])
        if random_shuffle:
            shuffle(self.image_files)
        #self.levels = [12,14,16,18,20,22,24,26,28,30,32,34,36,38,40] # 4um
        #self.levels = [5,8,11,14,17,20,23,26,29,32,35,38,41,44,47] # 6um
        self.model_input = 2
        self.distance = 6
        if self.distance == 4:
            self.levels = [16,18,20,22,24,26,28,30,32,34,36] # 4um
        elif self.distance == 6:
            self.levels = [8,11,14,17,20,23,26,29,32,35,38,41,44] # 6um
        elif self.distance == 8:
            self.levels = [6,10,14,18,22,26,30,34,38,42,46] # 8um
        #self.levels = [26,31,36,41,46,51]
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self,index):
        image,image_name = self.read_image(index)
        if self.labeled:
            label = self.read_label(image_name)
        else:
            label = None
        sample = {'image':image,'label':label}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def read_image(self,index):
        match_word = self.image_files[index][-28:][:20]
        level_1 = choice(self.levels)
        
        if self.distance == 4:
            level_0 = level_1 - 2
            level_2 = level_1 + 2
        elif self.distance == 6:
            level_0 = level_1 - 3
            level_2 = level_1 + 3
        elif self.distance == 8:
            
            if self.model_input == 3:
                level_0 = level_1 - 4
                level_2 = level_1 + 4
            elif self.model_input == 2:
                level_0 = level_1 -4
                level_2 = level_1
            elif self.model_input == 1:
                level_0 = level_1
                level_2 = level_1
            
        image_path_1 = self.images_dir + '/' + str(level_1)
        #image_path_0 = self.images_dir + '/' + str(level_0)
        image_path_2 = self.images_dir + '/' + str(level_2)
        img_name = image_path_1 + '/' + match_word + '_' +str(level_1).zfill(2) + '.tiff'
        
        image_1 = skimage.io.imread(image_path_1 + '/' + match_word + '_' +str(level_1).zfill(2) + '.tiff')
        #image_0 = skimage.io.imread(image_path_0 + '/' + match_word + '_' +str(level_0).zfill(2) + '.tiff')
        image_2 = skimage.io.imread(image_path_2 + '/' + match_word + '_' +str(level_2).zfill(2) + '.tiff')
        
        
        image_1[image_1>10000] = 10000
        #image_0[image_0>10000] = 10000
        image_2[image_2>10000] = 10000
        
        x = random.randint(0, image_1.shape[1]-image_size)
        y = random.randint(0,image_1.shape[0]-image_size)
        img_1 = image_1[y:y+image_size,x:x+image_size]
        #img_0 = image_0[y:y+image_size,x:x+image_size]
        img_2 = image_2[y:y+image_size,x:x+image_size]
        #img = np.dstack((img_0,img_1,img_2))
        img = np.dstack((img_1,img_2))
        
        if np.random.rand() > 0.8:
            scale = np.random.uniform(0.98,1.2)
            img = img*scale
        return img,img_name
    
    def read_label(self,filename):

        # 4um
        if self.distance == 4:
            if filename.endswith('_16.tiff'):
                return 0
            elif filename.endswith('_18.tiff'):
                return 1
            elif filename.endswith('_20.tiff'):
                return 2
            elif filename.endswith('_22.tiff'):
                return 3
            elif filename.endswith('_24.tiff'):
                return 4
            elif filename.endswith('_26.tiff'):
                return 5
            elif filename.endswith('_28.tiff'):
                return 6
            elif filename.endswith('_30.tiff'):
                return 7
            elif filename.endswith('_32.tiff'):
                return 8
            elif filename.endswith('_34.tiff'):
                return 9
            elif filename.endswith('_36.tiff'):
                return 10
        elif self.distance == 6:
            
            if filename.endswith('_08.tiff'):
                return 0
            elif filename.endswith('_11.tiff'):
                return 1
            elif filename.endswith('_14.tiff'):
                return 2
            elif filename.endswith('_17.tiff'):
                return 3
            elif filename.endswith('_20.tiff'):
                return 4
            elif filename.endswith('_23.tiff'):
                return 5
            elif filename.endswith('_26.tiff'):
                return 6
            elif filename.endswith('_29.tiff'):
                return 7
            elif filename.endswith('_32.tiff'):
                return 8
            elif filename.endswith('_35.tiff'):
                return 9
            elif filename.endswith('_38.tiff'):
                return 10
            elif filename.endswith('_41.tiff'):
                return 11
            elif filename.endswith('_44.tiff'):
                return 12
            
        elif self.distance == 8:
            if filename.endswith('_06.tiff'):
                return 0
            elif filename.endswith('_10.tiff'):
                return 1
            elif filename.endswith('_14.tiff'):
                return 2
            elif filename.endswith('_18.tiff'):
                return 3
            elif filename.endswith('_22.tiff'):
                return 4
            elif filename.endswith('_26.tiff'):
                return 5
            elif filename.endswith('_30.tiff'):
                return 6
            elif filename.endswith('_34.tiff'):
                return 7
            elif filename.endswith('_38.tiff'):
                return 8
            elif filename.endswith('_42.tiff'):
                return 9
            elif filename.endswith('_46.tiff'):
                return 10
        
class Defocus_dataset_1(Dataset):
    def __init__(self,image_dir,labeled,transform=None,random_shuffle=True):
        self.images_dir = image_dir
        self.infocuse_dir = image_dir + '/26'
        self.transform = transform
        self.labeled = labeled
        self.image_files = np.array([join(image_dir,file_name) for file_name in sorted(listdir(self.infocuse_dir)) if is_tiff(file_name)])
        if random_shuffle:
            shuffle(self.image_files)
        #self.levels = [12,14,16,18,20,22,24,26,28,30,32,34,36,38,40] # 4um
        #self.levels = [5,8,11,14,17,20,23,26,29,32,35,38,41,44,47] # 6um
        self.model_input = 2
        self.distance = 6
        if self.distance == 4:
            self.levels = [16,18,20,22,24,26,28,30,32,34,36] # 4um
        elif self.distance == 6:
            self.levels = [8,11,14,17,20,23,26,29,32,35,38,41,44] # 6um
        elif self.distance == 8:
            self.levels = [6,10,14,18,22,26,30,34,38,42,46] # 8um
        #self.levels = [26,31,36,41,46,51]
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self,index):
        image,image_name = self.read_image(index)
        if self.labeled:
            label = self.read_label(image_name)
        else:
            label = None
        sample = {'image':image,'label':label}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def read_image(self,index):
        match_word = self.image_files[index][-28:][:20]
        level_1 = choice(self.levels)
        
        if self.distance == 4:
            level_0 = level_1 - 2
            level_2 = level_1 + 2
        elif self.distance == 6:
            level_0 = level_1 - 3
            level_2 = level_1 + 3
        elif self.distance == 8:
            
            if self.model_input == 3:
                level_0 = level_1 - 4
                level_2 = level_1 + 4
            elif self.model_input == 2:
                level_0 = level_1 -4
                level_2 = level_1
            elif self.model_input == 1:
                level_0 = level_1
                level_2 = level_1
            
        image_path_1 = self.images_dir + '/' + str(level_1)
        #image_path_0 = self.images_dir + '/' + str(level_0)
        image_path_2 = self.images_dir + '/' + str(level_2)
        img_name = image_path_1 + '/' + match_word + '_' +str(level_1).zfill(2) + '.tiff'
        
        image_1 = skimage.io.imread(image_path_1 + '/' + match_word + '_' +str(level_1).zfill(2) + '.tiff')
        #image_0 = skimage.io.imread(image_path_0 + '/' + match_word + '_' +str(level_0).zfill(2) + '.tiff')
        #image_2 = skimage.io.imread(image_path_2 + '/' + match_word + '_' +str(level_2).zfill(2) + '.tiff')
        
        
        image_1[image_1>10000] = 10000
        #image_0[image_0>10000] = 10000
        #image_2[image_2>10000] = 10000
        
        x = random.randint(0, image_1.shape[1]-image_size)
        y = random.randint(0,image_1.shape[0]-image_size)
        img_1 = image_1[y:y+image_size,x:x+image_size]
        #img_0 = image_0[y:y+image_size,x:x+image_size]
        #img_2 = image_2[y:y+image_size,x:x+image_size]
        #img = np.dstack((img_0,img_1,img_2))
        #img = np.dstack((img_1,img_1))
        img = np.expand_dims(img_1,axis=2)
        if np.random.rand() > 0.8:
            scale = np.random.uniform(0.98,1.2)
            img = img*scale
        return img,img_name
    
    def read_label(self,filename):

        # 4um
        if self.distance == 4:
            if filename.endswith('_16.tiff'):
                return 0
            elif filename.endswith('_18.tiff'):
                return 1
            elif filename.endswith('_20.tiff'):
                return 2
            elif filename.endswith('_22.tiff'):
                return 3
            elif filename.endswith('_24.tiff'):
                return 4
            elif filename.endswith('_26.tiff'):
                return 5
            elif filename.endswith('_28.tiff'):
                return 6
            elif filename.endswith('_30.tiff'):
                return 7
            elif filename.endswith('_32.tiff'):
                return 8
            elif filename.endswith('_34.tiff'):
                return 9
            elif filename.endswith('_36.tiff'):
                return 10
        elif self.distance == 6:
            if filename.endswith('_08.tiff'):
                return 0
            elif filename.endswith('_11.tiff'):
                return 1
            elif filename.endswith('_14.tiff'):
                return 2
            elif filename.endswith('_17.tiff'):
                return 3
            elif filename.endswith('_20.tiff'):
                return 4
            elif filename.endswith('_23.tiff'):
                return 5
            elif filename.endswith('_26.tiff'):
                return 6
            elif filename.endswith('_29.tiff'):
                return 7
            elif filename.endswith('_32.tiff'):
                return 8
            elif filename.endswith('_35.tiff'):
                return 9
            elif filename.endswith('_38.tiff'):
                return 10
            elif filename.endswith('_41.tiff'):
                return 11
            elif filename.endswith('_44.tiff'):
                return 12
            
        elif self.distance == 8:
            if filename.endswith('_06.tiff'):
                return 0
            elif filename.endswith('_10.tiff'):
                return 1
            elif filename.endswith('_14.tiff'):
                return 2
            elif filename.endswith('_18.tiff'):
                return 3
            elif filename.endswith('_22.tiff'):
                return 4
            elif filename.endswith('_26.tiff'):
                return 5
            elif filename.endswith('_30.tiff'):
                return 6
            elif filename.endswith('_34.tiff'):
                return 7
            elif filename.endswith('_38.tiff'):
                return 8
            elif filename.endswith('_42.tiff'):
                return 9
            elif filename.endswith('_46.tiff'):
                return 10

class Normalizer(object):
    
    def __init__(self,mean=None,std = None):
        if mean == None:
            self.mean = np.array([[[0]]])
        else:
            self.mean = mean
        
        if std == None:
            self.std = np.array([[[10000-0]]])
        else:
            self.std = std
        
    def __call__(self,sample):
        image,label = sample['image'].astype(np.float32),sample['label']
        
            
        sample = {'image':((image-self.mean)/self.std),'label':label}
        return sample
    
class Augmenter(object):
    def __call__(self,sample,flip=0.5):
        image = sample['image'].astype(np.float32)
        if np.random.rand() < 0.5:
            image = image[:,::-1,:]
            sample['image'] = image
        if np.random.rand() < 0.5:
            sample['image'] = image[::-1,:,:]
        if sample['label'] == None:
            return {'image': torch.from_numpy(image.copy()).type(torch.FloatTensor).permute(2,0,1)}
        sample['label'] = np.array(sample['label'])
        #return {'image': torch.from_numpy(sample['image'].type(torch.FloatTensor).permute(2,0,1),'label':torch.from_numpy(sample['label']).type(torch.FloatTensor)}
        return {'image': torch.from_numpy(image.copy()).type(torch.FloatTensor).permute(2,0,1),'label':torch.from_numpy(sample['label']).type(torch.FloatTensor)}