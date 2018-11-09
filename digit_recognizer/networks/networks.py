# source activate py36gputorch041
# cd /mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/digit_recognizer/networks/
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

# class CNN(nn.Module):
#     def __init__(self, num_classes=10):
#         super(CNN,self).__init__()
#         self.layer1=nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2,stride=2))
#         self.fc = nn.Linear(7*7*32,num_classes)
#         self.softmax=nn.Sequential(
#             nn.Softmax())
#     def forward(self, x):
#         out=self.layer1(x)
#         out=self.layer2(out)
#         out=out.reshape(out.size(0), -1)
#         out=self.fc(out)
#         # out=self.softmax(out)
#         return out

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN,self).__init__()
        
        self.layer1=nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,stride=1,padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3,stride=1,padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64,128,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128,256,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.fc=nn.Sequential(
            nn.Linear(1024,512),
            nn.Linear(512,256),
            nn.Linear(256,10))
        
        self.softmax=nn.Sequential(
            nn.Softmax())
    
    def forward(self, x):
        out=self.layer1(x)
        out=self.layer2(out)
        # print("out",out.shape)
        # out torch.Size([100, 256, 1, 1])

        out=out.reshape(out.size(0),-1)
        out=self.fc(out)

        # out=self.softmax(out)

        return out
