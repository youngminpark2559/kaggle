# conda activate py36gputorch041
# cd /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/networks/
# rm e.l && python networks.py 2>&1 | tee -a e.l && code e.l

# ================================================================================
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
from sklearn import svm

# ================================================================================
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets,models,transforms
from pytorchcv.model_provider import get_model as ptcv_get_model

# ================================================================================
from src.networks import cbam as cbam

# ================================================================================
def init_weights(m):
  if type(m) == nn.Linear or\
     type(m) == nn.Conv2d or\
     type(m) == nn.ConvTranspose2d:
    torch.nn.init.xavier_uniform_(m.weight.data)
    # torch.nn.init.xavier_uniform_(m.bias)
    # m.bias.data.fill_(0.01)

# ================================================================================
def crop_and_concat(upsampled,bypass, crop=False):
  if crop:
    c=(bypass.size()[2]-upsampled.size()[2])//2
    bypass=F.pad(bypass,(-c,-c,-c,-c))
  return torch.cat((upsampled,bypass),1)

# ================================================================================
class Interpolate(nn.Module):
  def __init__(
    self,size=None,scale_factor=None,mode="bilinear",align_corners=True):
    super(Interpolate,self).__init__()
    self.interp=F.interpolate
    self.size=size
    self.scale_factor=scale_factor
    self.mode=mode
    self.align_corners=align_corners

  def forward(self,x):
    x=self.interp(
      x,size=self.size,scale_factor=self.scale_factor,
      mode=self.mode,align_corners=self.align_corners)
    return x

# ================================================================================
class Pretrained_ResNet152(nn.Module):

  def __init__(self):
    super(Pretrained_ResNet152,self).__init__()
    
    # c resnet152: pretrained resnet152
    self.resnet152=models.resnet152(pretrained=True)
    
    # ================================================================================
    # Freeze all parameters because you will not train (or edit) those pretrained parameters
    for param in self.resnet152.parameters():
      param.requires_grad=False

    # ================================================================================
    # print("self.resnet152",dir(self.resnet152))
    self.resnet152.fc=nn.Linear(in_features=2048,out_features=2,bias=True)

    # ================================================================================
    # Check edited model
    # print("self.resnet152",self.resnet152)
    # region edited resnet152
    # Sequential(
    #   (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #   (2): ReLU(inplace)
    #   (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    #   (4): Sequential(
    #     (0): Bottleneck(
    #       (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #       (downsample): Sequential(
    #         (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #         (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       )
    #     )
    #     (1): Bottleneck(
    #       (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (2): Bottleneck(
    #       (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #   )
    #   (5): Sequential(
    #     (0): Bottleneck(
    #       (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #       (downsample): Sequential(
    #         (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
    #         (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       )
    #     )
    #     (1): Bottleneck(
    #       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (2): Bottleneck(
    #       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (3): Bottleneck(
    #       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (4): Bottleneck(
    #       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (5): Bottleneck(
    #       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (6): Bottleneck(
    #       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (7): Bottleneck(
    #       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #   )
    #   (6): Sequential(
    #     (0): Bottleneck(
    #       (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #       (downsample): Sequential(
    #         (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
    #         (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       )
    #     )
    #     (1): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (2): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (3): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (4): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (5): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (6): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (7): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (8): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (9): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (10): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (11): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (12): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (13): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (14): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (15): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (16): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (17): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (18): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (19): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (20): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (21): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (22): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (23): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (24): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (25): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (26): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (27): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (28): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (29): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (30): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (31): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (32): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (33): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (34): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (35): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #   )
    #   (7): Sequential(
    #     (0): Bottleneck(
    #       (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #       (downsample): Sequential(
    #         (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
    #         (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       )
    #     )
    #     (1): Bottleneck(
    #       (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #     (2): Bottleneck(
    #       (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace)
    #     )
    #   )
    #   (8): AdaptiveAvgPool2d(output_size=(1, 1))
    #   (9): Linear(in_features=2048, out_features=1000, bias=True)
    #   (10): Linear(in_features=1000, out_features=2, bias=True)
    # )
    # endregion 

    # ================================================================================
    self.softmax_layer=nn.Softmax(dim=None)

  def forward(self,x):
    # print("x p",x.shape)

    # ================================================================================
    x=self.resnet152(x)
    # print("x p",x.shape)

    x=self.softmax_layer(x)

    return x

# ================================================================================
class Pretrained_ResNet50(nn.Module):

  def __init__(self):
    super(Pretrained_ResNet50,self).__init__()
    
    # c resnet50: pretrained resnet50
    self.resnet50=models.resnet50(pretrained=True)
    # region resnet50
    # print("self.resnet50",self.resnet50)
    # endregion 
    
    # ================================================================================
    # Freeze all parameters because you will not train (or edit) those pretrained parameters
    for param in self.resnet50.parameters():
      param.requires_grad=False

    # ================================================================================
    # print("self.resnet50",dir(self.resnet50))
    self.resnet50.fc=nn.Linear(in_features=2048,out_features=1,bias=True)

    # ================================================================================
    # Check edited model
    # print("self.resnet50",self.resnet50)
    # region edited resnet50

    # endregion 

    # ================================================================================
    # self.softmax_layer=nn.Softmax(dim=None)

  def forward(self,x):
    # print("x p",x.shape)

    # ================================================================================
    x=self.resnet50(x)
    # print("x p",x.shape)

    # x=self.softmax_layer(x)

    return x

class Pretrained_VGG16(nn.Module):

  def __init__(self):
    super(Pretrained_VGG16,self).__init__()
    
    # c vgg16: pretrained vgg16
    self.vgg16=models.vgg16(pretrained=True)
    # print("self.vgg16",self.vgg16)
    # print("self.vgg16",dir(self.vgg16))

    # ================================================================================
    # Freeze parameters of features and avgpool
    for param in self.vgg16.parameters():
      param.requires_grad=False
    
    # ================================================================================
    # print("self.resnet152",dir(self.resnet152))
    # self.resnet152.fc=nn.Linear(in_features=2048,out_features=2,bias=True)

    # print("self.vgg16.classifier",self.vgg16.classifier)
    # Sequential(
    #   (0): Linear(in_features=25088, out_features=4096, bias=True)
    #   (1): ReLU(inplace)
    #   (2): Dropout(p=0.5)
    #   (3): Linear(in_features=4096, out_features=4096, bias=True)
    #   (4): ReLU(inplace)
    #   (5): Dropout(p=0.5)
    #   (6): Linear(in_features=4096, out_features=1000, bias=True)
    # )
    classifier_list=list(self.vgg16.classifier)
    # print("classifier_list",classifier_list)
    # [Linear(in_features=25088, out_features=4096, bias=True), ReLU(inplace), Dropout(p=0.5), Linear(in_features=4096, out_features=4096, bias=True), ReLU(inplace), Dropout(p=0.5), Linear(in_features=4096, out_features=1000, bias=True)]

    classifier_list.append(nn.Linear(1000,1))

    self.vgg16.classifier=nn.Sequential(*classifier_list)
    
    # print("self.vgg16",self.vgg16)

    # ================================================================================
    self.softmax_layer=nn.Softmax(dim=None)
  
  def forward(self,x):
    # print("x p",x.shape)

    # ================================================================================
    x=self.vgg16(x)
    # print("x p",x.shape)

    # x=self.softmax_layer(x)

    return x

class Custom_Net(nn.Module):

  def __init__(self):
    super(Custom_Net,self).__init__()

    self.conv_1=nn.Sequential(
        nn.Conv2d(3,100,kernel_size=3,padding=1),
        nn.LeakyReLU(0.2),)
    self.conv_1.apply(init_weights)

    self.conv_2=nn.Sequential(
        nn.Conv2d(100,100,kernel_size=3,padding=1),
        nn.LeakyReLU(0.2),)
    self.conv_2.apply(init_weights)

    self.conv_3=nn.Sequential(
        nn.Conv2d(100,256,kernel_size=3,padding=0),
        nn.LeakyReLU(0.2),)
    self.conv_3.apply(init_weights)

    # ================================================================================
    self.conv_3_avgpool=nn.Sequential(
      nn.AvgPool2d(kernel_size=2))

    # ================================================================================
    self.conv_4=nn.Sequential(
        nn.Conv2d(256,256,kernel_size=3,padding=1),
        nn.LeakyReLU(0.2),)
    self.conv_4.apply(init_weights)

    self.conv_5=nn.Sequential(
        nn.Conv2d(256,256,kernel_size=3,padding=1),
        nn.LeakyReLU(0.2),)
    self.conv_5.apply(init_weights)

    self.conv_6=nn.Sequential(
        nn.Conv2d(256,256,kernel_size=3,padding=0),
        nn.LeakyReLU(0.2),)
    self.conv_6.apply(init_weights)
    
    # ================================================================================
    self.conv_6_avgpool=nn.Sequential(
      nn.AvgPool2d(kernel_size=2))
    
    # ================================================================================
    self.conv_7=nn.Sequential(
        nn.Conv2d(256,256,kernel_size=3,padding=1),
        nn.LeakyReLU(0.2),)
    self.conv_7.apply(init_weights)

    self.conv_8=nn.Sequential(
        nn.Conv2d(256,256,kernel_size=3,padding=1),
        nn.LeakyReLU(0.2),)
    self.conv_8.apply(init_weights)

    self.conv_9=nn.Sequential(
        nn.Conv2d(256,512,kernel_size=3,padding=0),
        nn.LeakyReLU(0.2),)
    self.conv_9.apply(init_weights)

    # ================================================================================
    self.conv_9_avgpool=nn.Sequential(
      nn.AvgPool2d(kernel_size=2))
    
    # ================================================================================
    self.conv_10=nn.Sequential(
        nn.Conv2d(512,512,kernel_size=3,padding=1),
        nn.LeakyReLU(0.2),)
    self.conv_10.apply(init_weights)

    self.conv_11=nn.Sequential(
        nn.Conv2d(512,512,kernel_size=3,padding=0),
        nn.LeakyReLU(0.2),)
    self.conv_11.apply(init_weights)
    
    # ================================================================================
    self.fc=nn.Sequential(
        nn.Linear(2048,1024,bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(1024,512,bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512,1,bias=True))
    self.fc.apply(init_weights)

    # ================================================================================
    self.softmax_layer=nn.Softmax(dim=None)
  
  def forward(self,x):
    # print("x p",x.shape)

    # ================================================================================
    o_conv1=self.conv_1(x)
    o_conv2=self.conv_2(o_conv1)
    o_conv3=self.conv_3(o_conv2)
    # print("o_conv3",o_conv3.shape)
    # torch.Size([11, 256, 90, 90])
    o_pool3=self.conv_3_avgpool(o_conv3)
    # torch.Size([11, 100, 48, 48])
    
    o_conv4=self.conv_4(o_pool3)
    o_conv5=self.conv_5(o_conv4)
    o_conv6=self.conv_6(o_conv5)
    o_pool6=self.conv_6_avgpool(o_conv6)
    # print("o_pool6",o_pool6.shape)
    # torch.Size([11, 256, 19, 19])

    o_conv7=self.conv_7(o_pool6)
    o_conv8=self.conv_8(o_conv7)
    o_conv9=self.conv_9(o_conv8)
    o_pool9=self.conv_9_avgpool(o_conv9)
    # print("o_pool9",o_pool9.shape)
    # torch.Size([11, 512, 6, 6])

    o_conv10=self.conv_10(o_pool9)
    o_conv11=self.conv_11(o_conv10)
    # print("o_conv11",o_conv11.shape)
    # torch.Size([11, 512, 2, 2])

    o_conv11_flat=o_conv11.view(o_conv11.shape[0],-1)
    # print("o_conv11_flat",o_conv11_flat.shape)
    # torch.Size([2048, 11])

    out=self.fc(o_conv11_flat)

    # out=self.softmax_layer(out)
    # print("out",out.shape)
    # torch.Size([11, 2])

    return out

# ================================================================================
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = cbam.CBAM( planes * 4, 16 )
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class ResNet50_CBAM(nn.Module):
    def __init__(self, layers,  network_type, num_classes, att_type=None):
        super(ResNet50_CBAM, self).__init__()

        self.inplanes = 64
        # self.network_type = network_type
        self.network_type = "ImageNet"
        self.block=Bottleneck
        network_type="ImageNet"
        
        # different model config between ImageNet and CIFAR 
        if network_type == "ImageNet":
            self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AvgPool2d(7)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        if att_type=='BAM':
            self.bam1 = BAM(64*self.block.expansion)
            self.bam2 = BAM(128*self.block.expansion)
            self.bam3 = BAM(256*self.block.expansion)
        else:
            self.bam1, self.bam2, self.bam3 = None, None, None

        self.layer1 = self._make_layer(self.block, 64,  layers[0], att_type=att_type)
        self.layer2 = self._make_layer(self.block, 128, layers[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(self.block, 256, layers[2], stride=2, att_type=att_type)
        self.layer4 = self._make_layer(self.block, 512, layers[3], stride=2, att_type=att_type)

        self.fc = nn.Linear(512 * self.block.expansion, num_classes)

        init.kaiming_normal_(self.fc.weight)

        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=att_type=='CBAM'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=att_type=='CBAM'))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.network_type == "ImageNet":
            x = self.maxpool(x)

        x = self.layer1(x)
        if not self.bam1 is None:
            x = self.bam1(x)

        x = self.layer2(x)
        if not self.bam2 is None:
            x = self.bam2(x)

        x = self.layer3(x)
        if not self.bam3 is None:
            x = self.bam3(x)

        x = self.layer4(x)

        if self.network_type == "ImageNet":
            x = self.avgpool(x)
        else:
            x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ================================================================================
class Pretrained_ResNet50_CBAM(nn.Module):
  def __init__(self):
    super(Pretrained_ResNet50_CBAM,self).__init__()
    
    self.model_conv=ptcv_get_model("cbam_resnet50",pretrained=True)
    # print("model_conv",model_conv)
    
    # ================================================================================
    # Freeze all parameters because you will not train (or edit) those pretrained parameters
    for param in self.model_conv.parameters():
      param.requires_grad=False

    # ================================================================================
    # @ Append layers

    # self.last_pool=nn.AdaptiveAvgPool2d((1, 1))
    self.last_fc=nn.Sequential(
      nn.Dropout(0.6),
      nn.Linear(in_features=1000,out_features=512,bias=True),
      nn.SELU(),
      nn.Dropout(0.8),
      nn.Linear(in_features=512,out_features=1,bias=True))

  def forward(self,x):
    # print("x p",x.shape)
    # x p torch.Size([40, 3, 224, 224])

    # ================================================================================
    x=self.model_conv(x)
    # print("x",x)

    x=self.last_fc(x)
    # print("x p",x.shape)
    # torch.Size([40, 1])

    return x

# ================================================================================
class Scikit_Learn_SVM():
  def __init__(self):
    SVM_clf=svm.SVC(gamma='scale')
    return SVM_clf

# ================================================================================




