import numpy as np
np.set_printoptions(threshold=np.nan)
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys,os,copy,argparse
import time,timeit,datetime
import glob,natsort
import cv2
from PIL import Image
from skimage.transform import resize
from skimage.restoration import (denoise_tv_chambolle,denoise_bilateral,
                                 denoise_wavelet,estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,\
                            precision_score,recall_score,fbeta_score,f1_score,roc_curve
from sklearn.model_selection import train_test_split
import scipy.misc
import scipy.optimize as opt
import scipy.special
from sklearn import svm
from skimage.viewer import ImageViewer
from random import shuffle
import gc
import Augmentor
import traceback
import subprocess
import csv
from xgboost import XGBClassifier

# ================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
from torch.autograd import gradcheck
from torch.optim.lr_scheduler import StepLR

# ================================================================================
from src.utils import utils_common as utils_common
from src.utils import utils_image as utils_image
from src.utils import utils_net as utils_net
from src.utils import utils_pytorch as utils_pytorch
from src.utils import utils_data as utils_data
from src.utils import utils_hook_functions as utils_hook_functions
from src.utils import utils_visualize_gradients as utils_visualize_gradients

from src.utils_for_dataset import custom_ds as custom_ds
from src.utils_for_dataset import custom_ds_test as custom_ds_test

from src.loss_functions import loss_functions_module as loss_functions_module
from src.metrics import metrics_module as metrics_module

from src.api_model import model_api_module as model_api_module
from src.api_text_file_path import text_file_path_api_module as text_file_path_api_module

from src.utils_analyzing_result import grad_cam as grad_cam

# ================================================================================
def train(args):
  k_fold=3
  epoch=int(args.epoch)
  batch_size=int(args.batch_size)
  # print("epoch",epoch)
  # print("batch_size",batch_size)
  # 9
  # 2
  
  # ================================================================================
  xgb=XGBClassifier(n_estimators=100)

  # ================================================================================
  text_file_instance=text_file_path_api_module.Path_Of_Text_Files(args)

  txt_of_train_data=text_file_instance.train_data
  # print("txt_of_train_data",txt_of_train_data)
  # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train_csv_path.txt

  # ================================================================================
  contents_of_txt,num_line=utils_common.return_path_list_from_txt(txt_of_train_data)
  # print("contents_of_txt",contents_of_txt)
  # ['/mnt/1T-5e7/mycodehtml/prac_data_science/kaggle/hivprogression/My_code/Data/training_data.csv']
  
  # ================================================================================

  train_data_wo_id_df=utils_data.load_HIV_csv_data(contents_of_txt[0])
  # print(train_data_wo_id_df.columns)
  # Index(['Resp', 'PR Seq', 'RT Seq', 'VL-t0', 'CD4-t0'], dtype='object')

  # print("train_data_wo_id_df",train_data_wo_id_df.shape)
  # (920, 5)

  # ================================================================================
  # freq_of_char_B,freq_of_char_H=utils_data.count_num_seq_containing_B_or_H(train_data_wo_id_df)
  # print("freq_of_char_B",freq_of_char_B)
  # print("freq_of_char_H",freq_of_char_H)
  # 1
  # 3

  # ================================================================================
  # B_mask_idx,H_mask_idx=utils_data.get_indices_containing_B_or_H(train_data_wo_id_df)
  # print("B_mask_idx",B_mask_idx)
  # print("H_mask_idx",H_mask_idx)
  # [25]
  # [43, 199, 843]

  # ================================================================================
  # train_data_wo_id_df=train_data_wo_id_df.drop(train_data_wo_id_df.index[[25,43,199,843]])
  # print("train_data_wo_id_df",train_data_wo_id_df.shape)
  # (916, 5)

  train_data_wo_id_df=train_data_wo_id_df.iloc[:-1,:]
 
  # ================================================================================
  # print("train_data_wo_id_df.columns",train_data_wo_id_df.columns)
  # Index(['Resp', 'PR Seq', 'RT Seq', 'VL-t0', 'CD4-t0'], dtype='object')

  # print("train_data_wo_id_df.columns",train_data_wo_id_df.shape)
  # (919, 5)

  train_data_wo_id_df=utils_data.resolve_label_imbalance_on_resp(train_data_wo_id_df)
  # print("train_data_wo_id_df",train_data_wo_id_df.shape)
  # (1466, 5)

  # ================================================================================
  # @ Length match in DNA sequence string

  # PR_Seq_old=train_data_wo_id_df.iloc[:,1]
  # RT_Seq_old=train_data_wo_id_df.iloc[:,2]

  # PR_Seq=utils_data.length_match_for_PR_Seq(PR_Seq_old)
  # # print("PR_Seq",PR_Seq)
  # # ['CCTCAAATCACTCTTTGGCAACGACCCCTCGTCCCAATAAGGATAGGGGGGCAACTAAAGGAAGCYCTATTAGATACAGGAGCAGATGATACAGTATTAGAAGACATGGAGTTGCCAGGAAGATGGAAACCAAAAATGATAGGGGGAATTGGAGGTTTTATCAAAGTAARACAGTATGATCAGRTACCCATAGAAATCTATGGACATAAAGCTGTAGGTACAGTATTAATAGGACCTACACCTGTCAACATAATTGGAAGAAATCTGTTGACTCAGCTTGGTTGCACTTTAAATTTY', 

  # RT_Seq=utils_data.length_match_for_RT_Seq(RT_Seq_old)
  # # print("RT_Seq",RT_Seq)
  # # ['CCCATTAGTCCTATTGAAACTGTACCAGTAAAGCTAAAGCCAGGAATGGATGGCCCAAAAGTTAAACAATGGCCATTGACAGAAGAAAAAATAAAAGCATTAGTAGAAATTTGYACAGAAATGGAAAAGGAAGGGAAAATTTCAAAAATTGGGCCTGAAAATCCATATAATACTCCAGTATTTGCCATAAAGAAAAAAGACAGTACTACATGGAGAAAATTAGTAGATTTCAGAGAACTTAATAAGAGAACTCAAGACTTCTGGGAAGTTCAAYTAGGAATACCACATCCCGCWGGGTTAAAAAAGAAYAAATCAGTAACAGTACTGGATGTGGGTGATGCATATTTCTCAGTTCCMTTAGATAAAGACTTCAGGAAGTATACTGCATTTACCATACCTAGTATAAACAATGAGACACCAGGGATTAGATATCAGTACAATGTGCTTCCACAGGGATGGAAAGGATCACCAGCAATATTCCAAAGTAGCATGACAAAAATCTTAGAGCCTTTTAGAAAACGAAATCCAGACATAGTTATCTACCAATACATGGATGATTTGTATGTAGGATCTGATTTRGAAATAGAACAGCATAGAACAAAAATAGAGGAACTGAGACAACATCTGTCAAGGTGGGGGTTTACCACACCAGACAAAAAACATCAGAAAGAACCTCCATTCCTTTGGATGGGCTATGAACTCCATCCTGATAAATGGACAGTACAGCCTATAGTTCTGCCAGAAAAAGATAGCTGGACTGTCAATGACATACAGAAGTTAGTGGGGAAGTTGAATTGGGCAAGTCAGATTTAYGCAGGGATTAAAGTAAAGCAATTATGTAAACTCCTTAGGGGGACCAAGKCACTAACAGAAATAATACCACTAACAAGAGAAGCAGAGCTAGAACTGGCAGAAAACAGGGAAATTCTAAAAGAACCAGTACATGGAGTGTATTATGATCCAACAAAAGACTTAATAGCAGAAATACAGAAGCAGGGGCAAGGC000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 

  # ================================================================================
  # @ Replace old with new

  # train_data_wo_id_df.iloc[:,1]=0
  # train_data_wo_id_df.iloc[:,1]=PR_Seq

  # train_data_wo_id_df.iloc[:,2]=0
  # train_data_wo_id_df.iloc[:,2]=RT_Seq

  # print("train_data_wo_id_df",train_data_wo_id_df.head(2))
  #    Resp  \
  # 0  0      
  # 1  0      

  #                                                                                                                                                                                                                                                                                                       PR Seq  \
  # 0  CCTCAAATCACTCTTTGGCAACGACCCCTCGTCCCAATAAGGATAGGGGGGCAACTAAAGGAAGCYCTATTAGATACAGGAGCAGATGATACAGTATTAGAAGACATGGAGTTGCCAGGAAGATGGAAACCAAAAATGATAGGGGGAATTGGAGGTTTTATCAAAGTAARACAGTATGATCAGRTACCCATAGAAATCTATGGACATAAAGCTGTAGGTACAGTATTAATAGGACCTACACCTGTCAACATAATTGGAAGAAATCTGTTGACTCAGCTTGGTTGCACTTTAAATTTY   
  # 1  CCTCAAATCACTCTTTGGCAACGACCCCTCGTCGCAATAAAGATAGGGGGGCAACTAAAGGAAGCTCTATTAGATACAGGAGCAGATGATACAGTATTAGAAGACATGGAATTGCCAGGAAGATGGAAACCAAAAATAATAGGGGGAATTGGAGGTTTTATCAAAGTAAGACAGTATGATCAGATACCCATAGAAATCTGTGGACATAAAGTTATAAGTACAGTATTAATAGGACCTACACCTGTCAACATAATTGGAAGAAATCTGATGACTCAGCTTGGTTGCACTTTAAATTTT   

  #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       RT Seq  \
  # 0  CCCATTAGTCCTATTGAAACTGTACCAGTAAAGCTAAAGCCAGGAATGGATGGCCCAAAAGTTAAACAATGGCCATTGACAGAAGAAAAAATAAAAGCATTAGTAGAAATTTGYACAGAAATGGAAAAGGAAGGGAAAATTTCAAAAATTGGGCCTGAAAATCCATATAATACTCCAGTATTTGCCATAAAGAAAAAAGACAGTACTACATGGAGAAAATTAGTAGATTTCAGAGAACTTAATAAGAGAACTCAAGACTTCTGGGAAGTTCAAYTAGGAATACCACATCCCGCWGGGTTAAAAAAGAAYAAATCAGTAACAGTACTGGATGTGGGTGATGCATATTTCTCAGTTCCMTTAGATAAAGACTTCAGGAAGTATACTGCATTTACCATACCTAGTATAAACAATGAGACACCAGGGATTAGATATCAGTACAATGTGCTTCCACAGGGATGGAAAGGATCACCAGCAATATTCCAAAGTAGCATGACAAAAATCTTAGAGCCTTTTAGAAAACGAAATCCAGACATAGTTATCTACCAATACATGGATGATTTGTATGTAGGATCTGATTTRGAAATAGAACAGCATAGAACAAAAATAGAGGAACTGAGACAACATCTGTCAAGGTGGGGGTTTACCACACCAGACAAAAAACATCAGAAAGAACCTCCATTCCTTTGGATGGGCTATGAACTCCATCCTGATAAATGGACAGTACAGCCTATAGTTCTGCCAGAAAAAGATAGCTGGACTGTCAATGACATACAGAAGTTAGTGGGGAAGTTGAATTGGGCAAGTCAGATTTAYGCAGGGATTAAAGTAAAGCAATTATGTAAACTCCTTAGGGGGACCAAGKCACTAACAGAAATAATACCACTAACAAGAGAAGCAGAGCTAGAACTGGCAGAAAACAGGGAAATTCTAAAAGAACCAGTACATGGAGTGTATTATGATCCAACAAAAGACTTAATAGCAGAAATACAGAAGCAGGGGCAAGGC000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000   
  # 1  CCCATTAGTCCTATTGAAACTGTACCAGTAAAATTAAAGCCAGGAATGGATGGCCCAAAAGTTAAACAATGGCCATTGACAGAAGAAAAAATAAAAGCATTAGTAGAAATTTGTACAGARATGGAAARGGARGGGAAAATTTCAAAAATTGGGCCTGAAAATCCATACAATACTCCGGTATTTGTCATAAAGAAAAAGGACAGTACTAAGTGGAGAAAAGTAGTAGATTTCAGAGAACTTAATAAAAGAACTCAAGACTTCTGGGAAGTTCAATTAGGGATACCACATCCCGCAGGGWTAAAAAAGAATAAATCAGTAACAGTATTGGATGTGGGTGATGCATACTTTTCAGTTCCCTTAGATGAAGACTTCAGGAAGTATACTGCATTTACCATACCCAGTACAAACAATGAGACACCAGGGATTAGATATCAGTACAATGTGCTTCCACAGGGATGGAAAGGATCACCAGCAATATTCCAAAGTAGCATGACAAAAATTTTAGAGCCTTTTAGAAAACAAAATCCAGACATAGTTATCTATCAATACGTGGATGATTTGTATGTAGGATCTGACTTAGAAATAGGGCAGCATAGAACAAAAATAGAGGAACTGAGACAACATCTGCTGAAGTGGGGATTGACCACACCAGACAAAAAAYATCAGAAAGAACCTCCATTTCGTTGGATGGGTTATGAACTCCATCCTGATAAMTGGACAGTACAGCCTATAGTGCTGCCAGAAAAAGACAGCTGGACTGTCAATGACATACAGAAGTTAGTGGGAAAATTAAATTGGGCAAGCCAGATTTACGCAGGGATTAAAGTAAAGCAATTATGTAAACTCCTTAGGGGAACCAAGGCACTAACWGATGTAATACCACTAACAAGAGAAGCAGAGCTAGAACTG000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000   

  #   VL-t0  CD4-t0  
  # 0  4.3    145     
  # 1  3.6    224     

  # ================================================================================
  # train_k,vali_k=utils_data.get_k_folds(train_data_wo_id_df)
  # print("train_k",train_k)
  # [array([[0,
  #       'CCTCAAATCACTCTTTGGCAACGACCCCTCGTCCCAATAAGGATAGGGGGGCAACTAAAGGAAGCYCTATTAGATACAGGAGCAGATGATACAGTATTAGAAGACATGGAGTTGCCAGGAAGATGGAAACCAAAAATGATAGGGGGAATTGGAGGTTTTATCAAAGTAARACAGTATGATCAGRTACCCATAGAAATCTATGGACATAAAGCTGTAGGTACAGTATTAATAGGACCTACACCTGTCAACATAATTGGAAGAAATCTGTTGACTCAGCTTGGTTGCACTTTAAATTTY',
  #       'CCCATTAGTCCTATTGAAACTGTACCAGTAAAGCTAAAGCCAGGAATGGATGGCCCAAAAGTTAAACAATGGCCATTGACAGAAGAAAAAATAAAAGCATTAGTAGAAATTTGYACAGAAATGGAAAAGGAAGGGAAAATTTCAAAAATTGGGCCTGAAAATCCATATAATACTCCAGTATTTGCCATAAAGAAAAAAGACAGTACTACATGGAGAAAATTAGTAGATTTCAGAGAACTTAATAAGAGAACTCAAGACTTCTGGGAAGTTCAAYTAGGAATACCACATCCCGCWGGGTTAAAAAAGAAYAAATCAGTAACAGTACTGGATGTGGGTGATGCATATTTCTCAGTTCCMTTAGATAAAGACTTCAGGAAGTATACTGCATTTACCATACCTAGTATAAACAATGAGACACCAGGGATTAGATATCAGTACAATGTGCTTCCACAGGGATGGAAAGGATCACCAGCAATATTCCAAAGTAGCATGACAAAAATCTTAGAGCCTTTTAGAAAACGAAATCCAGACATAGTTATCTACCAATACATGGATGATTTGTATGTAGGATCTGATTTRGAAATAGAACAGCATAGAACAAAAATAGAGGAACTGAGACAACATCTGTCAAGGTGGGGGTTTACCACACCAGACAAAAAACATCAGAAAGAACCTCCATTCCTTTGGATGGGCTATGAACTCCATCCTGATAAATGGACAGTACAGCCTATAGTTCTGCCAGAAAAAGATAGCTGGACTGTCAATGACATACAGAAGTTAGTGGGGAAGTTGAATTGGGCAAGTCAGATTTAYGCAGGGATTAAAGTAAAGCAATTATGTAAACTCCTTAGGGGGACCAAGKCACTAACAGAAATAATACCACTAACAAGAGAAGCAGAGCTAGAACTGGCAGAAAACAGGGAAATTCTAAAAGAACCAGTACATGGAGTGTATTATGATCCAACAAAAGACTTAATAGCAGAAATACAGAAGCAGGGGCAAGGC',
  #       4.3, 145],

  # print("train_data_wo_id_df.columns",train_data_wo_id_df.columns)
  # Index(['Resp', 'PR Seq', 'RT Seq', 'VL-t0', 'CD4-t0'], dtype='object')

  X=train_data_wo_id_df.iloc[:,0]
  y=train_data_wo_id_df.iloc[:,1:]

  y_train,y_test,X_train,X_test=train_test_split(X,y,test_size=0.2,random_state=42)
  # print("X_train",X_train.shape)
  # print("X_test",X_test.shape)
  # print("y_train",y_train.shape)
  # print("y_test",y_test.shape)
  # (1172, 4)
  # (294, 4)
  # (1172,)
  # (294,)

  # ================================================================================
  # c loss_list: list which will stores loss values to plot loss
  loss_list=[]
  f1_score_list=[]

  # ================================================================================
  if args.task_mode=="train": # If you're in train mode
  
    # ================================================================================
    single_trn_data_k=np.array(X_train)
    # print("single_trn_data_k",single_trn_data_k)
    # [['CCTCAGATCACTCTTTGGCAACGACCCCTCGTCACAATAAAGATAGGGGGGCAACTAAAGGAAGCTCTATTAGATACAGGAGCAGATGATACAGTAYTAGAAGAAATAAGTTTGCCAGGAAGATGGAAACCAAAAATGATAGGGGGAATTGGAGGWTTTATCAAAGTAAGACAGTATGATCAGGTGCCCATAGAAATCTGTGGACATAAAACTATAGGTACAGTATTAGTAGGACCTACACCTGTCAACATAATTGGAAGAAATCTGTTGACTCAGCTTGGGTGCACTTTAAATTTT'
    #   'CCCATTAGTCCTATTGAAACTGTACCAGTAAAATTAAAGCCAGGGATGGATGGCCCAAAAGTTAAACAATGGCCATTGACAGAAGAGAAAATAAAAGCATTARTAGAAATTTGTRCAGAAYTGGAAAAGGAMGGAAAAATTTCAAAAATTGGGCCTGAAAATCCATACAATACYCCARTATTTGCCATAAAKAARAAAAACAGTACTAAATGGAGAAAATTAGTAGATTTCAGAGAACTTAATAAGAGAACTCAAGACTTCTGGGAAGTTCAATTAGGAATACCACATCCCGCAGGGTTAAAAAAGAAYAAATCAGTAACAGTACTGGATGTGGGGGATGCATATTTTTCARTTCCCTTAGATAAAGAATTCAGGAAGTATACTGCATTTACCATACCTAGTATAAACAATGAGACACCAGGGGTTAGGTATCAGTACAATGTGCTTCCACAGGGATGGAAAGGRTCACCAGCAATATTCCAATGTAGCATGACAAAAATCTTAGAGCCTTTTAGAAAACAAAAYCCAGAYATAGTTATCTATCAATACATGGATGATTTGTATGTAGGATCTGACTTRGAAATAGGGCAGCATAGAACAAAAATAGAGGAACTGAGACAACATCTGTGGAGGTGGGGATTTTACACACCAGACAAHAAACAYCAGAAAGAACCTCCMTTCCTTTGGATGGGKTATGAACTCCATCYTGATAAATGGACAGTACAGCCTATAACGCTGCCAGAAAAGGACAGCTGGACTGTCAATGACATACAGAAGTTAGTGGGAAAATTRAATTGGGCAAGTCAGATYTACCCAGGGATCAAAGTAAGGCARTTATGTAAACTCCTTAGGGGAGCCAAAGCACTAACAGAAGTAGTACCACTAACAGAAGAAGCAGARYTAGAAYTGGCAGAAAACAGGGAGATTYTAAGAGAACCAGTGCATGGAGTGTATTATGACCCATCAAAAGACTTAATAGCAGAAATACAGAARCAGGGGCAAGGCCAATGGACATATCAAATTTATCAAGAGCCATTTAARAATCTGAARACAGGAAAATATGCAAGAACGAGGGGTACCCACACTAATGATGTAAAACAGTTAACAGAGGCAGTGCAAAAAATAGCCACAGAAAGCATAGTAATATGGGGAAAGACTCCTAARTTTAGATTACCTATACAGAAAGAAACATGGGAAACATGGTGGACAGAGTATTGGCAAGCCACCTGGATTCCTGARTGGGAGTTTGTKAATACCCCTCCCTTAGTGAAATTATGGTACCAGCTAGAGAAAGARCCCATAGTAGGAGCAGAAACTTTCTATGTAGATGGGGCAGCTAACAGGGAGACTAAAYTAGGAAAAGCAGGATATGTTACTAACAAAGGAAGACAAAAAGTKGTCTCYCTAACTGACRCAACAAATCAGCAGACTGAGTTACAAGCAATTTATCTAGCTTTGCAGGATTCGGGAYTA'
    #   5.5 145]
    
    single_trn_lbl_k=np.array(y_train)
    # print("single_trn_lbl_k",single_trn_lbl_k)
    # [0 0 0 0 1 0 1 1 1 1 0 0 0 0 0 1 1 0 1 1 1 0 0 0 0 0 1 1 1 0 0 0 0 0 1 0 1

    # ================================================================================
    PR_Seq=single_trn_data_k[:,0]
    # print("PR_Seq",PR_Seq)
    # ['CCTCAGATCACTCTTTGGCAACGACCCCTCGTCACAATAAAGATAGGGGGGCARCTAARGGAAGCTCTATTAGAYACAGGAGCAGATGACACAGTATTAGAAGAAATAAGTTTGCCAGGAAGATGGAAACCAAAAATGATAGGGGGAATTGGAGGTTTTATCAAAGTAAGACAGTATGATCAGATACCTGTAGAAATCTGTGGGCATAAAGCTATAGGTACAGTGTTAGTAGGACCTACACCTGTCAACATAATKGGAAGAAATCTGTTGACTCAGATTGGCTGCACTTTAAATTTT'

    PR_Seq_converted=utils_data.process_PR_Seq(PR_Seq)
    # print("PR_Seq_converted",PR_Seq_converted)
    # [[3, 3, 20, 3, 1, 7, 1, 20, 3, 1, 3, 20, 3, 20, 20, 20, 7, 7, 3, 1, 1, 3, 7, 1, 3, 3, 3, 3, 20, 3, 7, 20, 3, 1, 3, 1, 7, 20, 1, 1, 7, 7, 7, 20, 1, 7, 7, 7, 7, 7, 7, 3, 1, 1, 13, 20, 1, 1, 1, 7,

    PR_Seq_converted_df=pd.DataFrame(PR_Seq_converted)
    print("PR_Seq_converted_df",PR_Seq_converted_df.shape)
    # (1172, 297)

    PR_Seq_converted_df=PR_Seq_converted_df.fillna(value=0)
    # print("PR_Seq_converted_df",PR_Seq_converted_df)
    # print("PR_Seq_converted_df",PR_Seq_converted_df.shape)
    # (1172, 297)

    PR_Seq_converted_np=np.array(PR_Seq_converted_df)
    # print("PR_Seq_converted_np",PR_Seq_converted_np)
    
    lack=320-PR_Seq_converted_np.shape[1]

    PR_Seq_converted_np=np.pad(PR_Seq_converted_np,((0,0),(0,lack)),'constant')
    # print("PR_Seq_converted_np",PR_Seq_converted_np)
    # print("PR_Seq_converted_np",PR_Seq_converted_np.shape)
    # (1172, 320)

    # ================================================================================
    RT_Seq=single_trn_data_k[:,1]

    RT_Seq_converted=utils_data.process_RT_Seq(RT_Seq)
    # print("RT_Seq_converted",RT_Seq_converted)
    # [[3, 3, 3, 1, 20, 20, 1, 7, 20, 3, 3, 20, 1, 20, 20, 7, 1, 1, 1, 3, 20, 7, 20, 1, 3, 3, 1, 7, 20, 1, 1, 1, 7, 3, 20, 1, 1, 1, 7, 3, 3, 1, 7, 7, 1, 1, 20, 7, 7, 1, 20, 7, 7, 3, 3, 3, 1, 1, 1, 1,

    RT_Seq_converted_df=pd.DataFrame(RT_Seq_converted)
    # print("RT_Seq_converted_df",RT_Seq_converted_df.shape)
    # (612, 1479)

    RT_Seq_converted_df=RT_Seq_converted_df.fillna(value=0)
    # print("RT_Seq_converted_df",RT_Seq_converted_df)
    # print("RT_Seq_converted_df",RT_Seq_converted_df.shape)
    # (612, 1479)

    RT_Seq_converted_np=np.array(RT_Seq_converted_df)
    # print("RT_Seq_converted_np",RT_Seq_converted_np)
    # print("RT_Seq_converted_np",RT_Seq_converted_np.shape)
    # (1172, 1482)
    
    lack=1600-RT_Seq_converted_np.shape[1]

    RT_Seq_converted_np=np.pad(RT_Seq_converted_np,((0,0),(0,lack)),'constant')
    # print("RT_Seq_converted_np",RT_Seq_converted_np.shape)
    # (1172, 1600)

    # ================================================================================
    VL=single_trn_data_k[:,2]
    # norm_VL=utils_common.normalize_1D_arr(VL)
    # print("VL",VL)
    # [4.3 5.7 3.5 3.4 4.4 4.1 5.9 4.6 3.3 4.7 4.1 5.2 3.4 3.5 4.0 3.7 3.5 5.7

    # print("norm_VL",norm_VL.shape)
    # (613,)

    VL=VL[:,np.newaxis]
    VL_np=np.array(VL)
    # print("VL_np",VL_np)
    # [[4.3]
    #  [5.7]
    # print("VL_np",VL_np.shape)
    # (612, 1)
    
    # ================================================================================
    CD4=single_trn_data_k[:,3]
    # print("CD4",CD4)
    # [145 206 572 221 384 184 199 247 155 115 414 242 349 308 325 316 117 109
    
    # plt.subplot(1,2,1)
    # plt.plot(CD4)
    # plt.title('Before normalizing CD4')

    # norm_CD4=utils_common.normalize_1D_arr(CD4)
    # plt.subplot(1,2,2)
    # plt.plot(norm_CD4)
    # plt.title('After normalizing CD4')
    # plt.show()
    
    # print("norm_CD4",norm_CD4.shape)
    # (613,)

    CD4=CD4[:,np.newaxis]
    # print("CD4",CD4)

    CD4_np=np.array(CD4).astype("float16")
    # print("CD4_np",CD4_np)
    # [[145]
    #  [206]
    # print("CD4_np",CD4_np.shape)
    # (612, 1)

    # ================================================================================
    # print("PR_Seq_converted_np",PR_Seq_converted_np.shape)
    # # (612, 320)
    # print("RT_Seq_converted_np",RT_Seq_converted_np.shape)
    # # (612, 1600)
    # print("VL_npVL",VL_np.shape)
    # # (612, 1)
    # print("CD4_np",CD4_np.shape)
    # # (612, 1)

    # print("PR_Seq_converted_np",PR_Seq_converted_np[0,:])
    # # (612, 320)
    # print("RT_Seq_converted_np",RT_Seq_converted_np[0,:])
    # # (612, 1600)
    # print("VL_npVL",VL_np[0,0])
    # # (612, 1)
    # print("CD4_np",CD4_np[0,0])
    # # (612, 1)

    # ================================================================================
    final_trn_data=np.hstack((PR_Seq_converted_np,RT_Seq_converted_np,VL_np,CD4_np))
    # print("final_trn_data",final_trn_data.shape)
    # (1172, 1922)

    single_trn_lbl_k_np=np.array(single_trn_lbl_k).astype('float16')
    # print("single_trn_lbl_k_np",single_trn_lbl_k_np.shape)
    # (1172,)

    xgb.fit(final_trn_data,single_trn_lbl_k_np)
    # print('took_time_min',took_time_min)
    # 0:00:03.687858

    # ================================================================================
    # End of training  

    # ================================================================================
    # one_dummy_data_for_test=utils_data.get_one_dummy_data_for_test()
    # print("one_dummy_data_for_test",one_dummy_data_for_test.shape)

    # aa=xgb.predict([one_dummy_data_for_test])
    # print("aa",aa)
    # [0.]

    X_test=np.array(X_test)
    y_test=np.array(y_test)
    # print("X_test",X_test.shape)
    # # (294, 4)
    # print("y_test",y_test.shape)
    # # (294,)

    single_vali_data_k=X_test
    single_vali_lbl_k=y_test

    single_vali_lbl_k_np=np.array(single_vali_lbl_k).astype("float16")

    # ================================================================================
    # print("single_vali_data_k",single_vali_data_k.shape)
    # (305, 4)
    PR_Seq=single_vali_data_k[:,0]
    RT_Seq=single_vali_data_k[:,1]
    VL=single_vali_data_k[:,2]
    CD4=single_vali_data_k[:,3]

    # ================================================================================
    PR_Seq_converted=utils_data.process_PR_Seq(PR_Seq)
    # print("PR_Seq_converted",PR_Seq_converted)
    # [[3, 3, 20, 3, 1, 1, 1, 20, 3, 1, 3, 20, 3, 20, 20, 20, 7, 7, 3, 1, 1, 3, 7, 1, 3, 3, 3, 3, 20, 3, 7, 20, 3, 7, 3, 1, 1, 20, 1, 1, 1, 7, 1, 20, 1, 7, 7, 7, 7, 7, 7, 3, 1, 1, 3,

    PR_Seq_converted_df=pd.DataFrame(PR_Seq_converted)
    # print("PR_Seq_converted_df",PR_Seq_converted_df.shape)
    # (307, 297)

    PR_Seq_converted_df=PR_Seq_converted_df.fillna(value=0)
    # print("PR_Seq_converted_df",PR_Seq_converted_df)
    # print("PR_Seq_converted_df",PR_Seq_converted_df.shape)
    # (307, 297)

    PR_Seq_converted_np=np.array(PR_Seq_converted_df)
    # print("PR_Seq_converted_np",PR_Seq_converted_np)
    
    lack=320-PR_Seq_converted_np.shape[1]

    PR_Seq_converted_np=np.pad(PR_Seq_converted_np,((0,0),(0,lack)),'constant')
    # print("PR_Seq_converted_np",PR_Seq_converted_np)
    # print("PR_Seq_converted_np",PR_Seq_converted_np.shape)
    # (307, 320)

    # ================================================================================
    RT_Seq_converted=utils_data.process_RT_Seq(RT_Seq)
    # print("RT_Seq_converted",RT_Seq_converted)
    # [[3, 3, 3, 1, 20, 20, 1, 7, 20, 3, 3, 20, 1, 20, 20, 7, 1, 1, 1, 3, 20, 7, 20, 1, 3, 3, 1, 7, 20, 1, 1, 1, 7, 3, 20, 1, 1, 1, 7, 3, 3, 1, 7, 7, 1, 1, 20, 7, 7, 1, 20, 7, 7, 3, 3, 3, 1, 1, 1, 1,

    RT_Seq_converted_df=pd.DataFrame(RT_Seq_converted)
    # print("RT_Seq_converted_df",RT_Seq_converted_df.shape)
    # (612, 1479)

    RT_Seq_converted_df=RT_Seq_converted_df.fillna(value=0)
    # print("RT_Seq_converted_df",RT_Seq_converted_df)
    # print("RT_Seq_converted_df",RT_Seq_converted_df.shape)
    # (612, 1479)

    RT_Seq_converted_np=np.array(RT_Seq_converted_df)
    # print("RT_Seq_converted_np",RT_Seq_converted_np)
    
    lack=1600-RT_Seq_converted_np.shape[1]

    RT_Seq_converted_np=np.pad(RT_Seq_converted_np,((0,0),(0,lack)),'constant')

    # ================================================================================
    VL=VL[:,np.newaxis]
    VL_np=np.array(VL)
    # print("VL_np",VL_np)
    # [[4.3]
    #  [5.7]
    # print("VL_np",VL_np.shape)
    # (612, 1)
    
    # ================================================================================
    CD4=CD4[:,np.newaxis]
    # print("CD4",CD4)

    CD4_np=np.array(CD4).astype("float16")
    # print("CD4_np",CD4_np)
    # [[145]
    #  [206]
    # print("CD4_np",CD4_np.shape)
    # (612, 1)

    # ================================================================================
    final_vali_data=np.hstack((PR_Seq_converted_np,RT_Seq_converted_np,VL_np,CD4_np))
    final_vali_pred=xgb.predict(final_vali_data)
    # print("final_vali_pred",final_vali_pred)
    # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.

    # print("single_vali_lbl_k_np",single_vali_lbl_k_np)
    # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.

    b_c_mat=confusion_matrix(single_vali_lbl_k_np,final_vali_pred,labels=[0,1])
    # print("b_c_mat",b_c_mat)
    # [[125  21]
    #  [ 14 134]]

    # True Positive (Tumor pic is predicted as tumor)      False Negative (Tumor pic is predicted as non-tumor)
    # False Positive (Non-tumor pic is predicted as tumor) True Negative (Non-tumor pic is predicted as non-tumor)

    # ================================================================================
    report=classification_report(single_vali_lbl_k_np,final_vali_pred,target_names=['class Non tumor (neg)', 'class Tumor (pos)'])
    # print("Report",report)
    # Report
    #                        precision    recall  f1-score   support

    # class Non tumor (neg)       0.90      0.86      0.88       146
    #     class Tumor (pos)       0.86      0.91      0.88       148

    #             micro avg       0.88      0.88      0.88       294
    #             macro avg       0.88      0.88      0.88       294
    #          weighted avg       0.88      0.88      0.88       294

    # ================================================================================
    print("Accuracy_score",accuracy_score(single_vali_lbl_k_np,final_vali_pred))
    # Accuracy_score 0.8809523809523809

    print("Precision_score",precision_score(single_vali_lbl_k_np,final_vali_pred))
    # Precision_score 0.864516129032258

    print("Recall_score",recall_score(single_vali_lbl_k_np,final_vali_pred))
    # Recall_score 0.9054054054054054

    # print("fbeta_score",fbeta_score(y_true, y_pred, beta))
    
    print("F1_score",fbeta_score(single_vali_lbl_k_np,final_vali_pred,beta=1))
    # F1_score 0.8844884488448843

    # ================================================================================
    # @ ROC curve
    fpr,tpr,thresholds=roc_curve(single_vali_lbl_k_np,final_vali_pred)
    plt.plot(fpr,tpr,'o-',label="Binary classification")
    plt.title('Receiver Operating Characteristic-Curve')
    plt.show()

    # ================================================================================
    # @ 1-Fold evaluation

    # b_c_mat 
    # [[247   2]
    #  [  9  49]]
    
    # report
    #                        precision    recall  f1-score   support
    # class Non tumor (neg)       0.96      0.99      0.98       249
    #     class Tumor (pos)       0.96      0.84      0.90        58

    #             micro avg       0.96      0.96      0.96       307
    #             macro avg       0.96      0.92      0.94       307
    #          weighted avg       0.96      0.96      0.96       307

    # accuracy_score 0.9641693811074918
    # precision_score 0.9607843137254902
    # recall_score 0.8448275862068966
    # f1_score 0.8990825688073395

    # ================================================================================
    # @ 2-Fold evaluation

    # b_c_mat
    # [[243   0]
    #  [ 11  52]]
    
    # report
    #                        precision    recall  f1-score   support

    # class Non tumor (neg)       0.96      1.00      0.98       243
    #     class Tumor (pos)       1.00      0.83      0.90        63

    #             micro avg       0.96      0.96      0.96       306
    #             macro avg       0.98      0.91      0.94       306
    #          weighted avg       0.97      0.96      0.96       306

    # accuracy_score 0.9640522875816994
    # precision_score 1.0
    # recall_score 0.8253968253968254
    # f1_score 0.9043478260869565

    # ================================================================================
    # @ 3-Fold evaluation

    # b_c_mat
    # [[227  14]
    #  [ 44  21]]
    
    # report
    #                        precision    recall  f1-score   support

    # class Non tumor (neg)       0.84      0.94      0.89       241
    #     class Tumor (pos)       0.60      0.32      0.42        65

    #             micro avg       0.81      0.81      0.81       306
    #             macro avg       0.72      0.63      0.65       306
    #          weighted avg       0.79      0.81      0.79       306

    # accuracy_score 0.8104575163398693
    # precision_score 0.6
    # recall_score 0.3230769230769231
    # f1_score 0.42

  # elif args.task_mode=="submission":
  #   with torch.no_grad(): # @ Use network without calculating gradients
      
  #     sub_ds=custom_ds_test.custom_ds_Submission()
  #     print("sub_ds",sub_ds)

  #     sub_dl=torch.utils.data.DataLoader(
  #       dataset=sub_ds,batch_size=batch_size,shuffle=False,num_workers=3)
  #     print("sub_dl",sub_dl)

  #     # ================================================================================
  #     # @ c num_imgs_test: number of entire test images

  #     num_imgs_test=len(sub_ds)

  #     # ================================================================================
  #     # @ Create network and optimizer

  #     if args.train_method=="train_by_transfer_learning_using_resnet":
  #       model_api_instance=model_api_module.Model_API_class(args)

  #     # ================================================================================
  #     label_submission=pd.read_csv("/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/sample_submission.csv",encoding='utf8')
  #     base_names=label_submission.iloc[:,0].tolist()
  #     # print("base_names",base_names)

  #     # ================================================================================
  #     predicted_values=[]
  #     # @ Iterate all batches (batch1+batch2+...+batchn=entire images)
  #     for idx,data in enumerate(sub_dl):
  #       # print("idx",idx)
  #       # print("data",data)
  #       # 0
  #       # ['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/test/0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5.tif',
        
  #       imgs=data

  #       # ================================================================================
  #       test_imgs_list=[]
  #       for one_img_path in imgs:
  #         one_loaded_img=utils_image.load_img(one_img_path)
  #         # print("one_loaded_img",one_loaded_img.shape)
  #         # (96, 96, 3)

  #         one_loaded_img=resize(one_loaded_img,(224,224))

  #         test_imgs_list.append(one_loaded_img)

  #       # ================================================================================
  #       test_imgs_np=np.array(test_imgs_list).transpose(0,3,1,2)
        
  #       # @ If you want to use center (48,48) image from (96,96) image
  #       # test_imgs_np=test_imgs_np[:,:,24:72,24:72]
  #       # print("test_imgs_np",test_imgs_np.shape)
  #       # (11, 3, 48, 48)

  #       test_imgs_tc=Variable(torch.Tensor(test_imgs_np).cuda())
  #       # print("test_imgs_tc",test_imgs_tc.shape)
  #       # torch.Size([30, 3, 224, 224])

  #       # ================================================================================
  #       # @ Make predictions
  #       prediction=model_api_instance.gen_net(test_imgs_tc)
  #       # print("prediction",prediction)
  #       # tensor([[-2.0675],
  #       # ...
  #       #         [-1.2222]], device='cuda:0')

  #       sigmoid=torch.nn.Sigmoid()

  #       prediction_np=sigmoid(prediction).cpu().numpy()

  #       # ================================================================================
  #       # @ Make predicted labels

  #       prediction_np=np.where(prediction_np>0.5,1,0).squeeze()
  #       # print("prediction_np",prediction_np)
  #       # [0 0 1 0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 1 0 1 1 0 0]
  #       # print("lbls",lbls)
  #       # [1 0 0 0 0 1 0 0 1 0 0 1 1 0 1 0 1 0 0 0 0 0 0 0 1 0 1 1 1 0]

  #       predicted_values.extend(prediction_np)
     
  #     my_submission=pd.DataFrame({'id': base_names,'label': predicted_values})
  #     my_submission.to_csv('youngminpar2559_submission.csv',index=False)
