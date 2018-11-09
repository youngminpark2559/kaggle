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

# ======================================================================
# c train_d: train data csv file
train_d=util_files.load_csv_in_pd("./all/train.csv")

# c test_d: train data csv file
test_d=util_files.load_csv_in_pd("./all/test.csv")

train_X,train_y=util_common.prepare_dataset(train_d,test=False)
# train_X (42000, 28, 28)
# train_y (42000,)

test_X=util_common.prepare_dataset(test_d,test=True)
# test_X (28000, 28, 28)

# ======================================================================
if solver=="NN":
    predic_te=train.solve_by_CNN(train_X,train_y,test_X)
    # print("predic_te",predic_te)

    test_X_df=pd.DataFrame(
        test_X.reshape(test_X.shape[0],test_X.shape[1]*test_X.shape[2]))
    # (28000,784)

    # predic_te_df=pd.DataFrame(predic_te)
    # (28000,)

    test_X_df['label'] = predic_te

    cols = test_X_df.columns.tolist()
    cols.insert(0, cols.pop(-1))

    test_X_df=test_X_df[cols]
    # print("test_X_df",test_X_df.shape)
    # test_X_df (28000, 785)

    test_X_df.to_csv("processed.csv", sep=',')

    # c w_te_img: width of test images
    w_te_img=int(np.sqrt(test_X_df.shape[1]-1))
    # print("w_te_img",w_te_img)
    # w_te_img 28

    # c te_labels: labels of test images
    te_labels=np.array(test_X_df)[:,0].astype("uint8")
    # c te_imgs: images of test images
    te_imgs=np.array(test_X_df)[:,1:].reshape(-1,w_te_img,w_te_img)

    # c o_p_t_img: one prediced test image
    # for o_p_t_img in range(50):
    #     plt.imshow(te_imgs[o_p_t_img,:,:],cmap="gray")
    #     plt.title(te_labels[o_p_t_img])
    #     plt.show()

    s_sub=util_files.load_csv_in_pd("./all/sample_submission_original.csv")
    # print('s_sub.loc[:,"Label"]',s_sub.loc[:,"Label"].shape)
    # s_sub.loc[:,"Label"] (28000,)
    
    s_sub.loc[:,"Label"]=te_labels
    s_sub.to_csv("./all/sample_submission2.csv", sep=',')

    s_sub=util_files.load_csv_in_pd("./all/sample_submission2.csv")
    print("s_sub",s_sub.head())
