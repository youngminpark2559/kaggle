# conda activate py36gputorch041
# cd /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/utils/
# rm e.l && python utils_image.py 2>&1 | tee -a e.l && code e.l

# ================================================================================
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
from sklearn.model_selection import RepeatedKFold
import skimage
from skimage.morphology import square
from skimage.restoration import denoise_tv_chambolle
import timeit
import sys,os
import glob
import natsort 
from itertools import zip_longest # for Python 3.x
import math
import traceback

# ================================================================================
from src.networks import networks as networks

from src.utils import utils_image as utils_image

# ================================================================================
def get_file_list(path):
    file_list=glob.glob(path)
    file_list=natsort.natsorted(file_list,reverse=False)
    return file_list

# ================================================================================
def return_path_list_from_txt(txt_file):
    txt_file=open(txt_file, "r")
    read_lines=txt_file.readlines()
    num_file=int(len(read_lines))
    txt_file.close()
    return read_lines,num_file

# ================================================================================
def chunks(l,n):
  # For item i in range that is length of l,
  for i in range(0,len(l),n):
    # Create index range for l of n items:
    yield l[i:i+n]

# ================================================================================
def chunk_proteins_by_4C(path_of_protein_imgs):
  path_of_protein_imgs_li=[]
  for i in range(0,len(path_of_protein_imgs),4):
    one_protein_pic=path_of_protein_imgs[i:i+4]
    # print("one_protein_pic",one_protein_pic)
    # ['/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_blue.png\n',
    #  '/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_green.png\n',
    #  '/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_red.png\n',
    #  '/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_yellow.png\n']

    path_of_protein_imgs_li.append(one_protein_pic)
  
  return path_of_protein_imgs_li

# ================================================================================
def split_paths_of_images(path_of_protein_imgs_li,ninety_percent_portion):
  # c path_trn: divided paths (90%) for train
  path_trn=path_of_protein_imgs_li[:ninety_percent_portion]
  # print("path_trn",path_trn.shape)
  # (27964, 4)

  # c path_vali: divided paths (10%) for validation
  path_vali=path_of_protein_imgs_li[ninety_percent_portion:]
  # print("path_vali",path_vali.shape)
  # (3108, 4)

  # c num_imgs_of_path_trn: number of train images
  num_imgs_of_path_trn=len(path_trn)

  # c num_imgs_of_path_vali: number of validation images
  num_imgs_of_path_vali=len(path_vali)

  return path_trn,path_vali,num_imgs_of_path_trn,num_imgs_of_path_vali

# ================================================================================
def split_label_data(loaded_label_data_sorted_list,ninety_percent_portion):

  loaded_label_data_trn=loaded_label_data_sorted_list[:ninety_percent_portion]
  loaded_label_data_vali=loaded_label_data_sorted_list[ninety_percent_portion:]
  
  num_of_loaded_label_data_trn=len(loaded_label_data_trn)
  num_of_loaded_label_data_vali=len(loaded_label_data_vali)

  return loaded_label_data_trn,loaded_label_data_vali,num_of_loaded_label_data_trn,num_of_loaded_label_data_vali

# ================================================================================
def split_by_k_folds(train_data_wo_id_df):
  # K-Fold validation
  # - You split dataset into K chunks
  # - If you use 3-Fold validation like this chunk1/chunk2/chunk3,
  # (1) Train: chunk1/chunk2, validation: chunk3
  # (2) Train: chunk1/chunk3, validation: chunk2
  # (3) Train: chunk2/chunk3, validation: chunk1

  # - K-Fold validation is advantageous when you have imbalance dataset
  
  # - Code
  # from sklearn.model_selection import RepeatedKFold
  # splitter = RepeatedKFold(n_splits=3, n_repeats=1, random_state=0)
  # (1) It makes 3 folds.
  # (2) If n_repeats=2, it makes 6 folds

  # ================================================================================
  splitter=RepeatedKFold(n_splits=3,n_repeats=1,random_state=0)
  
  train_index_set=[]
  validation_index_set=[]
  for train_idx,vali_idx in splitter.split(train_data_wo_id_df):
    
    train_index_set.append(train_idx)
    # print("train_idx",train_idx)
    # [  0   3   4   6   7   9  11  13  15  16  17  19  21  22  23  24  25  26
    
    validation_index_set.append(vali_idx)
    # print("vali_idx",vali_idx)
    # [  1   2   5   8  10  12  14  18  20  27  30  31  34  37  39  40  45  48

    # print("train_idx",len(train_idx))
    # print("train_idx",len(vali_idx))
    # 613
    # 307
  
  # ================================================================================
  # print("train_index_set",len(train_index_set))
  # 3
  
  # print("validation_index_set",len(validation_index_set))
  # 3

  # print("train_index_set",train_index_set[0])
  # print("train_index_set",len(train_index_set[0]))
  # print("train_index_set",len(train_index_set[1]))
  # print("train_index_set",len(train_index_set[2]))

  # print("validation_index_set",len(validation_index_set))

  return train_index_set,validation_index_set

# ================================================================================
def screening_dataset(loaded_label_data_sorted_list,num_imgs,trn_pairs):
  loaded_label_data_sorted_np=np.array(loaded_label_data_sorted_list)

  # ================================================================================
  # @ 1. Check number of images on both cases
  # print("num_imgs",num_imgs)
  # 31072

  nb_labels=loaded_label_data_sorted_np[:,0].shape[0]
  # print("nb_labels",nb_labels)
  # 31072

  assert num_imgs==nb_labels,"You must satisfy num_imgs==nb_labels"

  # ================================================================================
  # @ 2. Check pair of images

  for one_pair in list(trn_pairs):
    # print("one_pair",one_pair)
    # (array([
    #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/41370f9c-bbbd-11e8-b2ba-ac1f6b6435d0_blue.png\n',
    #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/41370f9c-bbbd-11e8-b2ba-ac1f6b6435d0_green.png\n',
    #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/41370f9c-bbbd-11e8-b2ba-ac1f6b6435d0_red.png\n',
    #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/41370f9c-bbbd-11e8-b2ba-ac1f6b6435d0_yellow.png\n'],
    #   dtype='<U141'), 
    #  ['41370f9c-bbbd-11e8-b2ba-ac1f6b6435d0', '23 0'])

    # ================================================================================
    fn_from_one_protein=one_pair[0][0].replace("\n","").split("/")[-1].split("_")[0]
    # print("fn_from_one_protein",fn_from_one_protein)
    # bb8cb54e-bbb3-11e8-b2ba-ac1f6b6435d0

    fn_from_one_protein_label=one_pair[1][0]
    # print("fn_from_one_protein_label",fn_from_one_protein_label)
    # bb8cb54e-bbb3-11e8-b2ba-ac1f6b6435d0

    # ================================================================================
    assert fn_from_one_protein==fn_from_one_protein_label,'You must satisfy fn_from_one_protein==fn_from_one_protein_label'

    # ================================================================================
    # @ 3. Order of BGRY should be consistent in all images

    one_protein_img_paths_4C=one_pair[0]
    # print("one_protein_img_paths_4C",one_protein_img_paths_4C)
    # ['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/b552da46-bb9a-11e8-b2b9-ac1f6b6435d0_blue.png\n',
    #  '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/b552da46-bb9a-11e8-b2b9-ac1f6b6435d0_green.png\n',
    #  '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/b552da46-bb9a-11e8-b2b9-ac1f6b6435d0_red.png\n',
    #  '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/b552da46-bb9a-11e8-b2b9-ac1f6b6435d0_yellow.png\n']

    should_be_blue=one_protein_img_paths_4C[0].split("/")[-1].split("_")[-1].split(".")[0]
    should_be_green=one_protein_img_paths_4C[1].split("/")[-1].split("_")[-1].split(".")[0]
    should_be_red=one_protein_img_paths_4C[2].split("/")[-1].split("_")[-1].split(".")[0]
    should_be_yellow=one_protein_img_paths_4C[3].split("/")[-1].split("_")[-1].split(".")[0]
    # print("should_be_blue",should_be_blue)
    # blue
    # print("should_be_green",should_be_green)
    # green
    # print("should_be_red",should_be_red)
    # red
    # print("should_be_yellow",should_be_yellow)
    # yellow

    assert should_be_blue=="blue",'You must satisfy should_be_blue=="blue"'
    assert should_be_green=="green",'You must satisfy should_be_green=="green"'
    assert should_be_red=="red",'You must satisfy should_be_red=="red"'
    assert should_be_yellow=="yellow",'You must satisfy should_be_yellow=="yellow"'

# ================================================================================
def one_hot_label(batch_size,label_values):
  # print("label_values",label_values)
  # [[1], [1]]

  oh_label_arr=np.zeros((batch_size,28))
  # print("oh_label_arr",oh_label_arr.shape)
  # (2, 28)

  for i in range(batch_size):
    label_for_one=label_values[i]
    # print("label_for_one",label_for_one)
    # [1]

    for one_label in label_for_one:
      # print("one_label",one_label)
      oh_label_arr[i,int(one_label)]=1
  
  return oh_label_arr.astype("float16")

# ================================================================================
def one_hot_label_vali(batch_size,label_values):
  # print("label_values",label_values)
  # print("label_values",len(label_values))
  # [[1], [1]]

  oh_label_arr=np.zeros((batch_size,28))
  # print("oh_label_arr",oh_label_arr.shape)
  # (2, 28)

  for i in range(batch_size):
    label_for_one=label_values[i]
    # print("label_for_one",label_for_one)
    # [1]

    for one_label in label_for_one:
      # print("one_label",one_label)
      oh_label_arr[i,int(one_label)]=1
  
  return oh_label_arr.astype("float16")

# ================================================================================
def normalize_1D_arr(arr):
  min_val=np.min(arr)
  # print("min_val",min_val)
  # 0

  max_val=np.max(arr)
  # print("max_val",max_val)
  # 1219

  norm_arr=(arr-min_val)/(max_val-min_val)
  # print("norm_arr",norm_arr)
  # [0.11894995898277276 0.1689909762100082 0.4692370795734208

  norm_arr=norm_arr.astype("float16")
  # print("norm_arr",norm_arr)
  # [1.1896e-01 1.6895e-01 4.6924e-01 1.8127e-01 3.1494e-01 1.5100e-01

  return norm_arr

# ================================================================================
def divisorGenerator(n):
    large_divisors=[]
    for i in range(1,int(math.sqrt(n)+1)):
        if n%i==0:
            yield i
            if i*i!=n:
                large_divisors.append(n/i)
    for divisor in reversed(large_divisors):
        yield int(divisor)
# list(divisorGenerator(1024))

