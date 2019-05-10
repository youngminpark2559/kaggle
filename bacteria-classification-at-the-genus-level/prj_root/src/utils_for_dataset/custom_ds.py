
# /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/prj_root/src/utils_for_dataset/dataset_cgmit.py
# /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/prj_root/src/unit_test/Test_dataset_cgmit.py

# ================================================================================
import csv
import numpy as np
import pandas as pd
from random import shuffle

# ================================================================================
import torch
import torch.utils.data as data

# ================================================================================
from src.utils import utils_common as utils_common
from src.utils import utils_image as utils_image

# ================================================================================
class Custom_DS(data.Dataset):
  def __init__(self,single_train_k,single_train_lbl_k,args):
    # print("single_train_k",single_train_k)
    # ['/mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level/Train_Images/1907.png\n'

    # print("single_train_lbl_k",single_train_lbl_k)
    # ['salmonella' 'ecoli' 'staphylococus' 'staphylococus' 'listeria'

    zipped=list(zip(single_train_k,single_train_lbl_k))
    # print("zipped",zipped)
    # zipped [('/mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level/Train_Images/0004.png\n', array([4, 'ecoli'], dtype=object)), ('/mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level/Train_Images/0005.png\n', array([5, 

    shuffle(zipped)

    self.trn_pairs=zipped

    # ================================================================================
    # instance of argument 
    self.args=args

    # ================================================================================
    self.nb_trn_imgs=len(single_train_k)

  # ================================================================================
  def __len__(self):
    return self.nb_trn_imgs

  # ================================================================================
  def __getitem__(self,idx):
    one_pair=self.trn_pairs[idx]
    return one_pair

# ================================================================================
class Custom_DS_vali(data.Dataset):
  def __init__(self,single_vali_k,single_vali_lbl_k,args):
    # print("single_vali_k",single_vali_k)
    # ['/mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level/Train_Images/0035.png\n'
    #  '/mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level/Train_Images/0024.png\n']

    # print("single_vali_lbl_k",single_vali_lbl_k)
    # ['staphylococus' 'ecoli']

    zipped=list(zip(single_vali_k,single_vali_lbl_k))
    # print("zipped",zipped)
    # [['/mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level/Train_Images/0056.png\n'
    #   'ecoli']
    # ['/mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level/Train_Images/0023.png\n'
    #   'staphylococus']]

    # ================================================================================
    self.vali_pairs=zipped
    # print("self.vali_pairs",self.vali_pairs)

    # ================================================================================
    # instance of argument 
    self.args=args

    # ================================================================================
    self.nb_vali_imgs=len(single_vali_k)
    # print("self.nb_vali_imgs",self.nb_vali_imgs)
    # 10358

  # ================================================================================
  def __len__(self):
    return self.nb_vali_imgs

  # ================================================================================
  def __getitem__(self,idx):
    one_pair=self.vali_pairs[idx]
    return one_pair
