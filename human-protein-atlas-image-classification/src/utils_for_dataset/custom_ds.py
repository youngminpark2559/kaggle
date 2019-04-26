
# /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/prj_root/src/utils_for_dataset/dataset_cgmit.py
# /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/prj_root/src/unit_test/Test_dataset_cgmit.py

# @ Basic modules
import csv
import numpy as np
import pandas as pd
from random import shuffle
# @ PyTorch modules
import torch.utils.data as data
# @ src/utils
from src.utils import utils_common as utils_common
from src.utils import utils_image as utils_image

class Custom_DS(data.Dataset):
  def __init__(self,txt_containing_paths,txt_containing_labels,is_train,args):
    # print("txt_containing_paths",txt_containing_paths)
    # print("txt_containing_labels",txt_containing_labels)
    # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/Path_of_train_images.txt
    # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train.csv

    # ================================================================================
    # instance of argument 
    self.args=args

    # Is train in this step of creating dataset?
    self.is_train=is_train

    # ================================================================================
    path_of_protein_imgs=[]
    with open(txt_containing_paths) as f:
      lines=f.readlines()
      path_of_protein_imgs.extend(lines)
    # print("path_of_protein_imgs",path_of_protein_imgs)
    # ['/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_blue.png\n',
    #  '/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_green.png\n',
    #  '/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_red.png\n',
    #  '/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_yellow.png\n',
    #  '/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_blue.png\n',
    #  '/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_green.png\n',
    #  '/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_red.png\n',
    #  '/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_yellow.png\n',
    #  '/train/000a9596-bbc4-11e8-b2bc-ac1f6b6435d0_blue.png\n',
    #  '/train/000a9596-bbc4-11e8-b2bc-ac1f6b6435d0_green.png\n',
    #  '/train/000a9596-bbc4-11e8-b2bc-ac1f6b6435d0_red.png\n',
    #  '/train/000a9596-bbc4-11e8-b2bc-ac1f6b6435d0_yellow.png\n',
    #  '/train/000c99ba-bba4-11e8-b2b9-ac1f6b6435d0_blue.png\n',
    #  '/train/000c99ba-bba4-11e8-b2b9-ac1f6b6435d0_green.png\n',
    #  '/train/000c99ba-bba4-11e8-b2b9-ac1f6b6435d0_red.png\n',
    #  '/train/000c99ba-bba4-11e8-b2b9-ac1f6b6435d0_yellow.png\n',
    #  '/train/001838f8-bbca-11e8-b2bc-ac1f6b6435d0_blue.png\n',
    #  '/train/001838f8-bbca-11e8-b2bc-ac1f6b6435d0_green.png\n',
    #  '/train/001838f8-bbca-11e8-b2bc-ac1f6b6435d0_red.png\n',
    #  '/train/001838f8-bbca-11e8-b2bc-ac1f6b6435d0_yellow.png\n',

    # ================================================================================
    # print(len(path_of_protein_imgs))
    # 124288

    # print("len(path_of_protein_imgs)",len(path_of_protein_imgs)/4)
    # 31072
    
    # ================================================================================
    path_of_protein_imgs_li=utils_common.chunk_proteins_by_4C(path_of_protein_imgs)
    # print("path_of_protein_imgs_li",path_of_protein_imgs_li)
    # (31072, 4)

    # ================================================================================
    num_imgs=np.array(path_of_protein_imgs_li).shape[0]
    # print("num_imgs",num_imgs)
    # 31072

    # ================================================================================
    # c loaded_label_data: label data which is loaded from CSV file
    loaded_label_data=pd.read_csv(txt_containing_labels,encoding='utf8')
    # print("loaded_label_data",loaded_label_data)
    #                                          Id   Target
    # 0      00070df0-bbc3-11e8-b2bc-ac1f6b6435d0     16 0
    # 1      000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0  7 1 2 0

    # c loaded_label_data_sorted: sorted label data in ascending ID
    loaded_label_data_sorted=loaded_label_data.sort_values(by=["Id"],ascending=True)
    # print("loaded_label_data_sorted",loaded_label_data_sorted.head())
    #                                      Id   Target
    # 0  00070df0-bbc3-11e8-b2bc-ac1f6b6435d0     16 0
    # 1  000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0  7 1 2 0
    # print("loaded_label_data_sorted",loaded_label_data_sorted.shape)
    # (31072, 2)

    # c loaded_label_data_sorted_list: label into list
    loaded_label_data_sorted_list=loaded_label_data_sorted.iloc[:,:].values.tolist()
    # print("loaded_label_data_sorted_list",loaded_label_data_sorted_list)
    # [['00070df0-bbc3-11e8-b2bc-ac1f6b6435d0', '16 0'], 
    #  ['000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0', '7 1 2 0'], 
    #  ['000a9596-bbc4-11e8-b2bc-ac1f6b6435d0', '5'], 

    # ================================================================================
    # c ninety_percent_portion: 90% from entire images
    ninety_percent_portion=int(num_imgs*0.9)
    # print("ninety_percent_portion",ninety_percent_portion)
    # 27964

    # ================================================================================
    # @ Split paths of images

    path_trn,path_vali,num_imgs_of_path_trn,num_imgs_of_path_vali=\
      utils_common.split_paths_of_images(path_of_protein_imgs_li,ninety_percent_portion)
    
    # print("path_trn",len(path_trn))
    # 27964
    # print("path_vali",len(path_vali))
    # 3108
    # print("num_imgs_of_path_trn",num_imgs_of_path_trn)
    # 27964
    # print("num_imgs_of_path_vali",num_imgs_of_path_vali)
    # 3108

    # ================================================================================
    # @ Split labels

    loaded_label_data_trn,loaded_label_data_vali,\
    self.num_of_loaded_label_data_trn,self.num_of_loaded_label_data_vali=\
      utils_common.split_label_data(loaded_label_data_sorted_list,ninety_percent_portion)
    
    # print("np.array(loaded_label_data_trn)",np.array(loaded_label_data_trn).shape)
    # (27964, 2)
    # print("np.array(loaded_label_data_vali)",np.array(loaded_label_data_vali).shape)
    # (3108, 2)
    # print("self.num_of_loaded_label_data_trn",self.num_of_loaded_label_data_trn)
    # 27964
    # print("self.num_of_loaded_label_data_vali",self.num_of_loaded_label_data_vali)
    # 3108

    # ================================================================================
    # @ Create pairs of "train image path" and "label"

    self.trn_pairs=list(zip(path_trn,loaded_label_data_trn))
    shuffle(self.trn_pairs)

    # ================================================================================
    # @ Create pairs of validation image path and label

    self.vali_pairs=list(zip(path_vali,loaded_label_data_vali))

    # ================================================================================
    # @ Screening dataset

    utils_common.screening_dataset(loaded_label_data_sorted_list,num_imgs,self.trn_pairs)
    # print("took_time_min",took_time_min)
    # 0:00:00.073183

  # ================================================================================
  def __len__(self):

    if self.is_train==True:
      # Return number of entire train images
      return self.num_of_loaded_label_data_trn

    else:
      # Return number of entire test images
      return self.num_of_loaded_label_data_vali

  # ================================================================================
  def __getitem__(self,idx):

    if self.is_train==True: # Train mode
      # @ Return one pair of train image and label
      one_pair=self.trn_pairs[idx]
      return one_pair

    else:                   # Validation mode   
      # @ Return one pair of validation image and label
      one_pair=self.vali_pairs[idx]
      return one_pair
