# source activate py36gputorch041
# cd /mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/tgs-salt-identification-challenge/train/
# rm e.l && python train.py 2>&1 | tee -a e.l && code e.l

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
from xgboost import XGBClassifier,XGBRegressor
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
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# ======================================================================
currentdir="/mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/tgs-salt-identification-challenge/train"

network_dir="/mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/tgs-salt-identification-challenge/networks"
sys.path.insert(0,network_dir)

loss_function_dir="/mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/tgs-salt-identification-challenge/loss_functions"
sys.path.insert(0,loss_function_dir)

utils_dir="/mnt/1T-5e7/mycodehtml/prac_data_s/kaggle/tgs-salt-identification-challenge/utils"
sys.path.insert(0,utils_dir)

# import networks as networks
import loss_functions as loss_functions

import util_common as util_common
# import util_image as util_image
import util_nets as util_nets

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
epoch=100
batch_size=8

# ======================================================================
def solve_by_CNN(train_X,train_y,test_X):
    # c n_ei: number of entire train images
    n_ei=int(train_X.shape[0])
    # print("n_ei",n_ei)
    # n_ei 4000

    iteration=int(n_ei/batch_size)
    # print("iteration",iteration)
    # iteration 500

    gen_net,optimizer=util_nets.net_generator()

    # Iterates all epochs
    for one_ep in range(epoch):
        # Iterates all train images
        for itr in range(0,n_ei,batch_size):
            optimizer.zero_grad()

            # c one_b_tr_X: one batch of train X
            one_b_tr_X=train_X[itr:itr+batch_size,:,:].squeeze()
            # c one_b_tr_y: one batch of train y
            one_b_tr_y=train_y[itr:itr+batch_size].squeeze()
            # print("one_b_tr_X",one_b_tr_X.shape)
            # print("one_b_tr_y",one_b_tr_y.shape)
            # one_b_tr_X (8, 1, 128, 128)
            # one_b_tr_y (8, 1, 128, 128)

            # c train_X_tc: train X in torch
            train_X_tc=Variable(torch.Tensor(one_b_tr_X).unsqueeze(1).to(device))
            # print("train_X_tc",train_X_tc.shape)
            # train_X_tc torch.Size([100, 1, 28, 28])
            train_y_tc=Variable(torch.Tensor(one_b_tr_y).to(device))
            # print("train_y_tc",train_y_tc.shape)
            # train_y_tc torch.Size([8, 128, 128])
            # print("train_y_tc",train_y_tc)


            # c preds: predictions
            preds=gen_net(train_X_tc).squeeze()
            # print("preds",preds.shape)
            # preds torch.Size([8, 128, 128])

            pred_nums=torch.argmax(preds,dim=1)
            # print("pred_nums",pred_nums.shape)
            # pred_nums torch.Size([100])
            # print("pred_nums",pred_nums)

            # loss_v=loss_functions.ce_loss(preds,train_y_tc.long())
            loss_v=loss_functions.l1_loss(preds,train_y_tc)
            
            # loss_v=loss_functions.L1loss(pred_nums,train_y_tc)
            print("loss_v",loss_v)
            # loss_v tensor(2.3192, device='cuda:0', grad_fn=<NllLossBackward>)

            loss_v.backward()
            optimizer.step()

    # Save trained parameters' values at end of all epochs
    util_nets.save_checkpoint(
        {'state_dict': gen_net.state_dict(),
         'optimizer' : optimizer.state_dict()}, 
        checkpoint_path)
    print("Saved model at end of epoch")
    print("Train finished")

    with torch.no_grad():
        # c n_eti: number of entire test images
        n_eti=int(test_X.shape[0])
        # 28000

        # c iteration_te: iteration for test data
        iteration_te=int(n_eti/batch_size)
        # 280

        predic_te=[]
        for itr_te in range(0,n_eti,batch_size):
            one_b_te_X=test_X[itr_te:itr_te+batch_size,:,:]
            # print("one_b_te_X",one_b_te_X.shape)
            # one_b_te_X (100, 28, 28)

            test_X_tc=Variable(torch.Tensor(one_b_te_X).unsqueeze(1).to(device))
            # (100, 1, 28, 28)

            preds_te=gen_net(test_X_tc)
            print("preds_te",preds_te.shape)
            
            p_te_np=preds_te.detach().cpu().numpy()
            # print("p_te_np",p_te_np)

            pred_nums=np.argmax(p_te_np,axis=1)
            # print("pred_nums",pred_nums.shape)
            # print("pred_nums",pred_nums)

            predic_te.extend(pred_nums)

        return predic_te

def solve_by_xgboost(train_X,train_y,test_X,val=False):
    # c model: XGBClassifier model
    model=XGBClassifier()

    if val==True:
        # Split data
        train_X,val_X,train_y,val_y=util_common.split_dataset(train_X,train_y)

        # Reshape data
        rs_train_X,rs_test_X,rs_val_X=util_common.reshape_data_X(train_X,test_X,val_X)

        # Train XGBClassifier model with train dataset
        model.fit(rs_train_X,train_y)
    
        val_y_pred=model.predict(rs_val_X)

        print("accuracy_score(val_y,val_y_pred)",accuracy_score(val_y,val_y_pred))
        # accuracy_score(val_y,val_y_pred) 0.9292063492063493

        test_y_pred=model.predict(rs_test_X)
        # print("test_y_pred",test_y_pred.shape)
        # test_y_pred (28000,)

        return test_y_pred

    else:
        # Reshape data
        rs_train_X,rs_test_X=util_common.reshape_data_X(train_X,test_X)

        # Train XGBClassifier model with train dataset
        model.fit(rs_train_X,train_y)
    
        test_y_pred=model.predict(rs_test_X)
        # print("test_y_pred",test_y_pred.shape)
        # test_y_pred (28000,)

        return test_y_pred

def solve_by_ensenble(train_X,train_y,test_X,val=False):
    log_clf=LogisticRegression(random_state=42)
    rnd_clf=RandomForestClassifier(random_state=42)
    svm_clf=SVC(probability=True,random_state=42)

    voting_clf=VotingClassifier(
        estimators=[('lr',log_clf),('rf',rnd_clf),('svc',svm_clf)],
        # voting='soft')
        voting='hard')

    if val==True:
        # Split data
        train_X,val_X,train_y,val_y=util_common.split_dataset(train_X,train_y)
        
        # Reshape data
        rs_train_X,rs_test_X,rs_val_X=util_common.reshape_data_X(train_X,test_X,val_X)

        # Train XGBClassifier model with train dataset
        voting_clf.fit(rs_train_X,train_y)

        acc_sco=[]
        for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
            clf.fit(rs_train_X,train_y)
            val_y_pred=clf.predict(rs_val_X)
            acc_sco.append(accuracy_score(val_y,val_y_pred))
            # print(clf.__class__.__name__) 
        print("acc_sco",acc_sco)
    
        val_y_pred=voting_clf.predict(rs_val_X)

        print("accuracy_score(val_y,val_y_pred)",accuracy_score(val_y,val_y_pred))
        # accuracy_score(val_y,val_y_pred) 0.9292063492063493

        test_y_pred=voting_clf.predict(rs_test_X)
        # print("test_y_pred",test_y_pred.shape)
        # test_y_pred (28000,)

        return test_y_pred

    else:
        # Reshape data
        rs_train_X,rs_test_X=util_common.reshape_data_X(train_X,test_X)

        # Train XGBClassifier model with train dataset
        voting_clf.fit(rs_train_X,train_y)
    
        test_y_pred=voting_clf.predict(rs_test_X)
        # print("test_y_pred",test_y_pred.shape)
        # test_y_pred (28000,)

        return test_y_pred
