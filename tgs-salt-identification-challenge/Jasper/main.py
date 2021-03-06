# source activate py36gputorch041
# cd /mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/tgs-salt-identification-challenge/
# rm e.l && python main.py 2>&1 | tee -a e.l && code e.l

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import sys,os
from PIL import Image
import random
from skimage.transform import resize
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)

# ======================================================================
currentdir="/mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/tgs-salt-identification-challenge/train"

network_dir="/mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/tgs-salt-identification-challenge/networks"
sys.path.insert(0,network_dir)

loss_function_dir="/mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/tgs-salt-identification-challenge/loss_functions"
sys.path.insert(0,loss_function_dir)

utils_dir="/mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/tgs-salt-identification-challenge/utils"
sys.path.insert(0,utils_dir)

train_dir="/mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/tgs-salt-identification-challenge/train"
sys.path.insert(0,train_dir)

import loss_functions as loss_functions

import util_files as util_files
import util_common as util_common
import util_nets as util_nets

import train as train

# ======================================================================
if torch.cuda.is_available():
    device="cuda:0"
else:
    device="cpu"

# device=torch.device(device)
device=torch.device("cuda:0")

# ======================================================================
checkpoint_path="/mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/tgs-salt-identification-challenge/train/checkpoint.pth.tar"

# ======================================================================
solver="NN"

# ======================================================================
# Inspect data

ids= ['1f1cc6b3a4','5b7c160d0d','6c40978ddf','7dfdf6eeb8','7e5a6e5013']
plt.figure(figsize=(20,10))

for j, img_name in enumerate(ids):
    q = j+1
    
    img=Image.open('./input/train/images/' + img_name + '.png')
    img=np.array(img)

    img_mask=Image.open('./input/train/masks/' + img_name + '.png')
    img_mask=np.array(img_mask)
    
    plt.subplot(1,2*(1+len(ids)),q*2-1)
    plt.imshow(img,cmap="gray")
    plt.subplot(1,2*(1+len(ids)),q*2)
    plt.imshow(img_mask,cmap="gray")
plt.show()

path_train="/mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/tgs-salt-identification-challenge/input/train"
train_ids=util_files.get_data_ids(path_train)

path_test="/mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/tgs-salt-identification-challenge/input/test"
test_ids=util_files.get_data_ids(path_test)

# Create placeholder array which you will use after resize (101,101,3) image to (128,128,1) image
X_train=util_common.create_ph_arr((len(train_ids),128,128,1),"uint8")
Y_train=util_common.create_ph_arr((len(train_ids),128,128,1),"bool")
Y_train=Y_train.astype("float32")

X_train,Y_train=util_common.rs_tr_img(train_ids,path_train,X_train,Y_train)

X_test=np.zeros((len(test_ids),128,128,1),dtype=np.uint8)

for n,id_ in enumerate(test_ids):
    path=path_test
    te_img=Image.open(path+'/images/'+id_)
    te_img=np.array(te_img)[:,:,1]
    te_x=resize(te_img,(128,128,1),mode='constant',preserve_range=True)
    X_test[n]=te_x

# ======================================================================
if solver=="NN":
    predic_te=train.solve_by_CNN(X_train/255.0,Y_train/255.0,X_test/255.0)
