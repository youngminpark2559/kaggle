from PIL import Image
import PIL.ImageOps
import scipy.misc
import cv2
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
import timeit
import sys
import glob
from skimage.transform import resize

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable,Function
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms

# utils_dir="/mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/tgs-salt-identification-challenge/utils"
# sys.path.insert(0,utils_dir) 
# import utils

def l1_loss(pred_img,gt_img):
    summed=torch.sum(torch.abs(pred_img-gt_img))
    # diff_img=torch.abs(pred_img-gt_img)
    return summed

def ce_loss(pred,gt):
    loss=nn.CrossEntropyLoss()
    output=loss(pred,gt)
    return output

def L1loss(pred,gt):
    loss = nn.L1Loss()
    output=loss(pred,gt)
    return output