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
from torchvision import transforms,utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
logging.getLogger().setLevel(logging.INFO)

# U net

def dual_conv(in_channel,out_channel):
    conv = nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size=3),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(out_channel, out_channel, kernel_size=3),
                         nn.ReLU(inplace=True),)
    return conv

def crop_tensor(target_tensor,tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:,:,delta:tensor_size-delta,delta:tensor_size-delta]

class Unet(nn.Module):
    
    def __init__(self,in_channel):
        super(Unet,self).__init__()
        # Left
        self.dwn_conv1 = dual_conv(in_channel,64)
        self.dwn_conv2 = dual_conv(64,128)
        self.dwn_conv3 = dual_conv(128,256)
        self.dwn_conv4 = dual_conv(256,512)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        # Right
        self.trans1 = nn.ConvTranspose2d(512, 256, kernel_size = 2, stride = 2)
        self.up_conv1 = dual_conv(512,256)
        self.trans2 = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2)
        self.up_conv2 = dual_conv(256,128)
        self.trans3 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2)
        self.up_conv3 = dual_conv(128,64)
        # output layer
        self.out = nn.Conv2d(64, 1, kernel_size=1)
        self.fc1 = nn.Linear(1*36*36, 512)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(512,15)
        
    def forward(self,image):
        x1 = self.dwn_conv1(image)
        x2 = self.maxpool(x1)
        x3 = self.dwn_conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.dwn_conv3(x4)
        x6 = self.maxpool(x5)
        x7 = self.dwn_conv4(x6)
        
        # forward on right
        x = self.trans1(x7)
        y = crop_tensor(x,x5)
        x = self.up_conv1(torch.cat([x,y],1))
        
        x = self.trans2(x)
        y = crop_tensor(x,x3)
        x = self.up_conv2(torch.cat([x,y],1))
        
        x = self.trans3(x)
        y = crop_tensor(x, x1)
        x = self.up_conv3(torch.cat([x,y],1))
        
        x = self.out(x)
        x = x.view(-1,1*36*36)
        x  = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class VGGBlock(nn.Module):
    def __init__(self, in_channels,middle_channels,out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3,padding=0)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        return out

class VGGBlock_same(nn.Module):
    def __init__(self, in_channels,middle_channels,out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3,padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        return out
    
class UNet(nn.Module):
    def __init__(self,input_channels=3):
        super().__init__()
        nb_filter = [32,64,128,256,512]
        
        self.pool = nn.MaxPool2d(2,2)
        self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        #self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        
        #self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_2 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_3 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        
        self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        
        self.fc1 = nn.Linear(1*36*36, 512)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(512,11)
        
    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        #x4_0 = self.conv4_0(self.pool(x3_0))
        
        x2_1 = self.conv2_1(torch.cat([crop_tensor(self.up(x3_0), x2_0), self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([crop_tensor(self.up(x2_1), x1_0), self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([crop_tensor(self.up(x1_2), x0_0), self.up(x1_2)], 1))
        
        
        output = self.final(x0_3)
        output = output.view(-1,1*36*36)
        out  = F.relu(self.fc1(output))
        out = self.dropout(out)
        out = self.fc2(out)
        return out
    
    
class NestedUNet(nn.Module):
    def __init__(self,input_channels=3,deep_supervision=False):
        super().__init__()
        nb_filter = [32,64,128,256,512]
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        
        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        
        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        
        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        
        self.fc1 = nn.Linear(1*36*36, 512)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(512,11)
        
    def forward(self,input):
        
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([crop_tensor(self.up(x1_0),x0_0),self.up(x1_0)],1))
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([crop_tensor(self.up(x2_0),x1_0),self.up(x2_0)],1))
        x0_2 = self.conv0_2(torch.cat([crop_tensor(self.up(x1_1),x0_0),crop_tensor(self.up(x1_1), x0_1),self.up(x1_1)],1))
        
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([crop_tensor(self.up(x3_0), x2_0), self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([crop_tensor(self.up(x2_1),x1_0),crop_tensor(self.up(x2_1), x1_1),self.up(x2_1)],1))
        x0_3 = self.conv0_3(torch.cat([crop_tensor(self.up(x1_2),x0_0),crop_tensor(self.up(x1_2),x0_1),crop_tensor(self.up(x1_2),x0_2),self.up(x1_2)],1))
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
        else:
            output = self.final(x0_3)
        output = output.view(-1,1*36*36)
        out  = F.relu(self.fc1(output))
        out = self.dropout(out)
        out = self.fc2(out)
        return out
    
class Unet_3plus(nn.Module):
    def __init__(self,input_channels = 3):
        super().__init__()
        nb_filter = [32,64,128,256,512]
        self.pool = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(4,4)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        # encoder
        self.conv1 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv2 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv3 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv4 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        
        self.CatChannels = nb_filter[0]
        self.CatBlocks = 4
        self.UpChannels = self.CatChannels*self.CatBlocks
        
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(self.CatChannels)
        self.bnL = nn.BatchNorm2d(self.UpChannels) 
        
        self.conv1_3 = nn.Conv2d(nb_filter[0],self.CatChannels,3,padding=1)
        self.conv2_3 = nn.Conv2d(nb_filter[1],self.CatChannels,3,padding=1)
        self.conv3_3 = nn.Conv2d(nb_filter[2],self.CatChannels,3,padding=1)
        self.conv4_3 = nn.Conv2d(nb_filter[3],self.CatChannels,3,padding=1)
        self.conv3R = nn.Conv2d(self.UpChannels,self.UpChannels,3,padding=1)
        
        self.conv1_2 = nn.Conv2d(nb_filter[0],self.CatChannels,3,padding=1)
        self.conv2_2 = nn.Conv2d(nb_filter[1],self.CatChannels,3,padding=1)
        self.conv3_2 = nn.Conv2d(nb_filter[2],self.CatChannels,3,padding=1)
        self.conv4_2 = nn.Conv2d(nb_filter[3],self.CatChannels,3,padding=1)
        self.conv2R = nn.Conv2d(self.UpChannels,self.UpChannels,3,padding=1)
        
        self.conv1_1 = nn.Conv2d(nb_filter[0],self.CatChannels,3,padding=1)
        self.conv2_1 = nn.Conv2d(128,self.CatChannels,3,padding=1)
        self.conv3_1 = nn.Conv2d(128,self.CatChannels,3,padding=1)
        self.conv4_1 = nn.Conv2d(256,self.CatChannels,3,padding=1)
        self.conv1R = nn.Conv2d(self.UpChannels,self.UpChannels,3,padding=1)
        
        self.final = nn.Conv2d(128, 1, kernel_size=1)
        
        self.fc1 = nn.Linear(1*64*64, 512)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(512,11)
        
    def forward(self,input):
        # Encoder
        h1 = self.conv1(input)
        h2 = self.conv2(self.pool(h1))
        h3 = self.conv3(self.pool(h2))
        h4 = self.conv4(self.pool(h3))
        
        h1_3 = self.relu(crop_tensor(self.up(h4),self.conv1_3(self.pool4(h1))))
        h2_3 = self.relu(crop_tensor(self.up(h4),self.conv2_3(self.pool(h2))))
        h3_3 = self.relu(crop_tensor(self.up(h4),self.conv3_3(h3)))
        h4_3 = self.relu(self.conv4_3(self.up(h4)))
        h3_R = self.relu(self.bnL(self.conv3R(torch.cat((h1_3,h2_3,h3_3,h4_3),1))))
        
        h1_2 = self.relu(crop_tensor(self.up(h3_R),self.conv1_2(self.pool(h1))))
        h2_2 = self.relu(crop_tensor(self.up(h3_R),self.conv2_2(h2)))
        h3_2 = self.relu(self.conv3_2(self.up(h3_R)))
        h4_2 = self.relu(self.conv4_2(self.up2(h4)))
        h2_R = self.relu(self.bnL(self.conv2R(torch.cat((h1_2,h2_2,h3_2,h4_2),1))))
        
        h1_1 = self.relu(crop_tensor(self.up(h2_R),self.conv1_1(h1)))
        h2_1 = self.relu(self.conv2_1(self.up(h2_R)))
        h3_1 = self.relu(self.conv3_1(self.up2(h3_R)))
        h4_1 = self.relu(self.conv4_1(self.up3(h4)))
        h4_R =  self.relu(self.bnL(self.conv1R(torch.cat((h1_1,h2_1,h3_1,h4_1),1))))
        
        out = self.final(h4_R)
        out = out.view(-1,1*64*64)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class Unet3(nn.Module):
    def __init__(self,input_channels = 3):
        super().__init__()
        nb_filter = [64,128,256,512]
        self.pool = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(4,4)
        self.pool8 = nn.MaxPool2d(8,8)
        self.relu = nn.ReLU(inplace=True)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        
        self.CatChannels = nb_filter[0]
        self.CatBlocks = 4
        self.UpChannels = self.CatChannels*self.CatBlocks
        
        self.bnL = nn.BatchNorm2d(self.UpChannels) 
        # encoder
        self.conv1 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv2 = VGGBlock_same(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv3 = VGGBlock_same(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv4 = VGGBlock_same(nb_filter[2], nb_filter[3], nb_filter[3])
        
        #
        self.conv1_3 = nn.Conv2d(nb_filter[0], self.CatChannels, 3, padding=1)
        self.conv2_3 = nn.Conv2d(nb_filter[1], self.CatChannels, 3, padding=1)
        self.conv3_3 = nn.Conv2d(nb_filter[2], self.CatChannels, 3, padding=1)
        self.conv4_3 = nn.Conv2d(nb_filter[3], self.CatChannels, 3, padding=1)
        self.conv3R = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)
        
        self.conv1_2 = nn.Conv2d(nb_filter[0], self.CatChannels, 3, padding=1)
        self.conv2_2 = nn.Conv2d(nb_filter[1], self.CatChannels, 3, padding=1)
        self.conv3_2 = nn.Conv2d(nb_filter[2], self.CatChannels, 3, padding=1)
        self.conv4_2 = nn.Conv2d(nb_filter[3], self.CatChannels, 3, padding=1)
        self.conv2R = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)
        
        self.conv1_1 = nn.Conv2d(nb_filter[0], self.CatChannels, 3, padding=1)
        self.conv2_1 = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.conv3_1 = nn.Conv2d(nb_filter[2], self.CatChannels, 3, padding=1)
        self.conv4_1 = nn.Conv2d(nb_filter[3], self.CatChannels, 3, padding=1)
        self.conv1R = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  
        
        self.final1 = nn.Conv2d(self.UpChannels,16,3,padding=1)
        self.final2 = nn.Conv2d(16, 1, 3,padding=1)
        #self.final3 = nn.Conv2d(8, 1, 3,padding=1)
        self.fc1 = nn.Linear(1*30*30, 512)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(512,11)
        
    def forward(self, input):
        h1 = self.conv1(input)
        h2 = self.conv2(self.pool(h1))
        h3 = self.conv3(self.pool(h2))
        h4 = self.conv4(self.pool(h3))
        
        h1_3 = self.relu(self.conv1_3(self.pool4(h1)))
        h2_3 = self.relu(self.conv2_3(self.pool(h2)))
        h3_3 = self.relu(self.conv3_3(h3))
        h4_3 = self.relu(self.conv4_3(self.up(h4)))
        h3_R = self.relu(self.bnL(self.conv3R(torch.cat((h1_3,h2_3,h3_3,h4_3),1))))
        
        h1_2 = self.relu(self.conv1_2(self.pool(h1)))
        h2_2 = self.relu(self.conv2_2((h2)))
        h3_2 = self.relu(self.conv3_2(self.up(h3_R)))
        h4_2 = self.relu(self.conv4_2(self.up2(h4)))
        h2_R = self.relu(self.bnL(self.conv2R(torch.cat((h1_2,h2_2,h3_2,h4_2),1))))
        
        h1_1 = self.relu(self.conv1_1(h1))
        h2_1 = self.relu(self.conv2_1(self.up(h2_R)))
        h3_1 = self.relu(self.conv3_1(self.up2(h3_R)))
        h4_1 = self.relu(self.conv4_1(self.up3(h4)))
        h4_R = self.relu(self.bnL(self.conv1R(torch.cat((h1_1,h2_1,h3_1,h4_1),1))))
        
        out = self.final1(h4_R)
        out = self.pool(out)
        out = self.relu(out)
        out = self.final2(out)
        out = self.pool(out)
        out = self.relu(out)
        # fully connected layer
        out = out.view(-1,1*30*30)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out
    
    
class selfmade_net(nn.Module):
    def __init__(self):
        super(selfmade_net,self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding = 1,padding_mode = 'reflect')
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(2,stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1,padding_mode = 'reflect')
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1,padding_mode = 'reflect')
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding = 1,padding_mode = 'reflect')
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256*8*8,1024)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(1024,13)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(x)
        x = x.view(-1,256*8*8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
if __name__ == "__main__":
    
    image = torch.rand(10,2,128,128)
    model = selfmade_net()
    out = model(image)
    print(out.shape)