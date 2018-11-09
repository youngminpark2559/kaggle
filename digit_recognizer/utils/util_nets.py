# source activate py36gputorch041
# cd /mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/digit_recognizer/utils/
# rm e.l && python utils_image.py 2>&1 | tee -a e.l && code e.l

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
from sklearn.feature_extraction import image
import timeit
import sys
import glob
import natsort 
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

# ======================================================================
currentdir = "/mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/digit_recognizer/utils"
network_dir="/mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/digit_recognizer/networks"
sys.path.insert(0,network_dir)

import networks as networks

# ======================================================================
if torch.cuda.is_available():
    device="cuda:0"
else:
    device="cpu"

# device=torch.device(device)
device=torch.device("cuda:0")

# ======================================================================
def get_file_list(path):
    file_list=glob.glob(path)
    file_list=natsort.natsorted(file_list,reverse=False)
    return file_list

def print_network(net,struct=False):
    """
    Args
      net: created network
      struct (False): do you want to see structure of entire network?
    Print
      Structure of entire network
      Total number of parameters of network
    """
    if struct==True:
        print(net)
    
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    print('Total number of parameters: %d' % num_params)

def save_checkpoint(state, filename):
    torch.save(state, filename)

def net_generator():
    gen_net=networks.CNN().to(device)
    optimizer=torch.optim.Adam(
        gen_net.parameters(),lr=0.001)
    
    # optimizer=torch.optim.SGD(direct_intrinsic_net.parameters(),lr=0.01,momentum=0.9)

    # --------------------------------------------------
    print_network(gen_net)

    # --------------------------------------------------
    checkpoint = torch.load(
        "/mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/digit_recognizer/train/checkpoint.pth.tar")

    gen_net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    return gen_net,optimizer
