# source activate py36gputorch041
# cd /mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/digit_recognizer/
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
import sys
from PIL import Image
from skimage.transform import resize
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)

# ======================================================================
currentdir="/mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/digit_recognizer/train"

network_dir="/mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/digit_recognizer/networks"
sys.path.insert(0,network_dir)

loss_function_dir="/mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/digit_recognizer/loss_functions"
sys.path.insert(0,loss_function_dir)

utils_dir="/mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/digit_recognizer/utils"
sys.path.insert(0,utils_dir)

train_dir="/mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/digit_recognizer/train"
sys.path.insert(0,train_dir)

# import networks as networks
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
checkpoint_path="/mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/digit_recognizer/train/checkpoint.pth.tar"

# ======================================================================
solver="NN"
# solver="xgboost"
# solver="ensenble"

# ======================================================================
# c train_d: train data csv file
train_d=util_files.load_csv_in_pd("./all/train.csv")

# c test_d: train data csv file
test_d=util_files.load_csv_in_pd("./all/test.csv")

train_X,train_y=util_common.prepare_dataset(train_d,test=False)
# train_X=train_X[:300,:,:]
# train_y=train_y[:300]
# train_X (42000, 28, 28)
# train_y (42000,)

test_X=util_common.prepare_dataset(test_d,test=True)
# test_X (28000, 28, 28)

# ======================================================================
if solver=="NN":
    predic_te=train.solve_by_CNN(train_X/255.0,train_y,test_X/255.0)
    create_res_csv(predic_te,test_X,solver)

elif solver=="xgboost":
    # test_y_pred=train.solve_by_xgboost(
    #     train_X,train_y,test_X,val=True)
    test_y_pred=train.solve_by_xgboost(
        train_X,train_y,test_X,val=False)
    util_files.create_res_csv(test_y_pred,test_X,solver)

elif solver=="ensenble":
    # test_y_pred=train.solve_by_ensenble(train_X,train_y,test_X,val=True)
    test_y_pred=train.solve_by_ensenble(train_X,train_y,test_X,val=False)
    util_files.create_res_csv(test_y_pred,test_X,solver)

