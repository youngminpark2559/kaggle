# source activate py36gputorch041
# cd /mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/tgs-salt-identification-challenge/networks/
# rm e.l && python networks.py 2>&1 | tee -a e.l && code e.l

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import sys
import time
import os
import copy
import glob
import cv2
import natsort 
from PIL import Image
from skimage.transform import resize
import scipy.misc

# ======================================================================
if torch.cuda.is_available():
    device="cuda:0"
else:
    device="cpu"

# device=torch.device(device)
device=torch.device("cuda:0")

# ======================================================================
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class UNet(nn.Module):
    def __init__(self, num_classes=10):
        super(UNet,self).__init__()

        self.layer1=nn.Sequential(
            nn.Conv2d(1,8,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU())

        self.layer2=nn.Sequential(
            nn.Conv2d(8,8,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU())
        
        self.layer3=nn.Sequential(
            nn.MaxPool2d(kernel_size=2))
        
        self.layer4=nn.Sequential(
            nn.Conv2d(8,16,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())
        
        self.layer5=nn.Sequential(
            nn.Conv2d(16,16,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())
        
        self.layer6=nn.Sequential(
            nn.MaxPool2d(kernel_size=2))

        self.layer7=nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        
        self.layer8=nn.Sequential(
            nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        
        self.layer9=nn.Sequential(
            nn.MaxPool2d(kernel_size=2))

        self.layer10=nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer11=nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer12=nn.Sequential(
            nn.MaxPool2d(kernel_size=2))

        self.layer13=nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.layer14=nn.Sequential(
            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.layer15=nn.Sequential(
            nn.ConvTranspose2d(128,64,kernel_size=2,stride=2,padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer16=nn.Sequential(
            nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer17=nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer18=nn.Sequential(
            nn.ConvTranspose2d(64,32,kernel_size=2,stride=2,padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.layer19=nn.Sequential(
            nn.Conv2d(64,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        
        self.layer20=nn.Sequential(
            nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.layer21=nn.Sequential(
            nn.ConvTranspose2d(32,16,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU())
        
        self.layer22=nn.Sequential(
            nn.Conv2d(48,16,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())
        
        self.layer23=nn.Sequential(
            nn.Conv2d(16,16,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer24=nn.Sequential(
            nn.ConvTranspose2d(16,8,kernel_size=4,stride=4,padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU())

        self.layer25=nn.Sequential(
            nn.Conv2d(16,8,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU())
        
        self.layer26=nn.Sequential(
            nn.Conv2d(8,8,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU())
        
        self.layer27=nn.Sequential(
            nn.Conv2d(8,1,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid())

    def forward(self, x):
        c1=self.layer1(x)
        # print("c1",c1.shape)
        # c1 torch.Size([8, 8, 128, 128])
        c1=self.layer2(c1)
        # print("c1",c1.shape)
        # c1 torch.Size([8, 8, 128, 128])
        p1=self.layer3(c1)
        # print("p1",p1.shape)
        # p1 torch.Size([8, 8, 64, 64])

        c2=self.layer4(p1)
        # print("c2",c2.shape)
        # c2 torch.Size([8, 16, 64, 64])
        c2=self.layer5(c2)
        # print("c2",c2.shape)
        # c2 torch.Size([8, 16, 64, 64])
        p2=self.layer6(c2)
        # print("p2",p2.shape)
        # p2 torch.Size([8, 16, 32, 32])

        c3=self.layer7(p2)
        # print("c3",c3.shape)
        # c3 torch.Size([8, 32, 32, 32])
        c3=self.layer8(c3)
        # print("c3",c3.shape)
        # c3 torch.Size([8, 32, 32, 32])
        p3=self.layer9(c3)
        # print("p3",p3.shape)
        # p3 torch.Size([8, 32, 16, 16])

        c4=self.layer10(p3)
        # print("c4",c4.shape)
        # c4 torch.Size([8, 64, 16, 16])
        c4=self.layer11(c4)
        # print("c4",c4.shape)
        # c4 torch.Size([8, 64, 16, 16])
        p4=self.layer12(c4)
        # print("p4",p4.shape)
        # p4 torch.Size([8, 64, 8, 8])

        c5=self.layer13(p4)
        # print("c5",c5.shape)
        # c5 torch.Size([8, 128, 8, 8])
        c5=self.layer14(c5)
        # print("c5",c5.shape)
        # c5 torch.Size([8, 128, 8, 8])

        u6=self.layer15(c5)
        # print("u6",u6.shape)
        # u6 torch.Size([8, 64, 16, 16])
        # print("c4",c4.shape)
        # c4 torch.Size([8, 64, 16, 16])
        u6=torch.cat((u6,c4),dim=1)
        # print("u6",u6.shape)
        # u6 torch.Size([8, 128, 16, 16])

        c6=self.layer16(u6)
        # print("c6",c6.shape)
        # c6 torch.Size([8, 64, 16, 16])
        c6=self.layer17(c6)
        # print("c6",c6.shape)
        # c6 torch.Size([8, 64, 16, 16])

        u7=self.layer18(c6)
        # print("u7",u7.shape)
        # u7 torch.Size([8, 32, 32, 32])
        u7=torch.cat((u7,c3),dim=1)
        # print("u7",u7.shape)
        # u7 torch.Size([8, 64, 32, 32])
        c7=self.layer19(u7)
        # print("c7",c7.shape)
        # c7 torch.Size([8, 32, 32, 32])
        c7=self.layer20(c7)
        # print("c7",c7.shape)
        # c7 torch.Size([8, 32, 32, 32])

        u8=self.layer21(c7)
        # print("u8",u8.shape)
        # u8 torch.Size([8, 16, 32, 32])
        # c3 torch.Size([8, 32, 32, 32])
        u8=torch.cat((u8,c3),dim=1)
        # print("u8",u8.shape)
        # u8 torch.Size([8, 48, 32, 32])
        c8=self.layer22(u8)
        # print("c8",c8.shape)
        # c8 torch.Size([8, 16, 32, 32])
        c8=self.layer23(c8)
        # print("c8",c8.shape)
        # c8 torch.Size([8, 16, 32, 32])

        u9=self.layer24(c8)
        # print("u9",u9.shape)
        # u9 torch.Size([8, 8, 128, 128])
        # c1 torch.Size([8, 8, 128, 128])
        u9=torch.cat((u9,c1),dim=1)
        # print("u9",u9.shape)
        # u9 torch.Size([8, 16, 128, 128])
        c9=self.layer25(u9)
        # print("c9",c9.shape)
        # c9 torch.Size([8, 8, 128, 128])
        c9=self.layer26(c9)
        # print("c9",c9.shape)
        # c9 torch.Size([8, 8, 128, 128])

        outputs=self.layer27(c9)
        # print("outputs",outputs.shape)
        # outputs torch.Size([8, 1, 128, 128])

        return outputs
