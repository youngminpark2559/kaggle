
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

class Dataset_Tumor(data.Dataset):
  def __init__(self,txt_containing_paths,txt_containing_labels,is_train,args):
    # print("txt_containing_paths",txt_containing_paths)
    # print("txt_containing_labels",txt_containing_labels)
    # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/tumor_trn.txt
    # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_labels.csv
    
    # instance of argument 
    self.args=args

    # Is train in this step of creating dataset?
    self.is_train=is_train

    # ================================================================================
    path_of_imgs_tumor=[]
    with open(txt_containing_paths) as f:
      lines=f.readlines()
      path_of_imgs_tumor.extend(lines)
    # print("path_of_imgs_tumor",path_of_imgs_tumor)
    # ['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/00/00001b2b5609af42ab0ab276dd4cd41c3e7745b5.tif\n'

    # ================================================================================
    num_imgs=len(path_of_imgs_tumor)

    # ================================================================================
    # c label_tumor: label data which is loaded from CSV file
    label_tumor=pd.read_csv(txt_containing_labels,encoding='utf8')

    # c label_tumor_sorted: sorted label data in ascending ID
    label_tumor_sorted=label_tumor.sort_values(by=["id"], ascending=True)
    # print("label_tumor_sorted",label_tumor_sorted.head())
    #                                               id  label
    # 151577  00001b2b5609af42ab0ab276dd4cd41c3e7745b5      1
    # 16166   000020de2aa6193f4c160e398a8edea95b1da598      0

    # c label_tumor_sorted_list: label into list
    label_tumor_sorted_list=label_tumor_sorted.iloc[:,:].values.tolist()
    # print("label_tumor_sorted_list",label_tumor_sorted_list)
    # [['00001b2b5609af42ab0ab276dd4cd41c3e7745b5', 1], 
    #  ['000020de2aa6193f4c160e398a8edea95b1da598', 0],
    #  ...

    # ================================================================================
    # c ninety_percent_portion: 90% from entire images
    ninety_percent_portion=int(num_imgs*0.9)
    # print("ninety_percent_portion",ninety_percent_portion)
    # 198022

    # ================================================================================
    # @ Split image paths

    # c path_trn: divided paths (90%) for train
    path_trn=path_of_imgs_tumor[:ninety_percent_portion]

    # c path_vali: divided paths (10%) for validation
    path_vali=path_of_imgs_tumor[ninety_percent_portion:]

    # c num_imgs_of_path_trn: number of train images
    num_imgs_of_path_trn=len(path_trn)

    # c num_imgs_of_path_vali: number of validation images
    num_imgs_of_path_vali=len(path_vali)
    # print("self.num_imgs_of_path_trn",self.num_imgs_of_path_trn)
    # print("self.num_imgs_of_path_vali",self.num_imgs_of_path_vali)
    # 198022
    # 22003

    # ================================================================================
    # @ Split labels

    # c label_tumor_trn: labels for train images
    label_tumor_trn=label_tumor_sorted_list[:ninety_percent_portion]
    
    # c label_tumor_vali: labels for validation images
    label_tumor_vali=label_tumor_sorted_list[ninety_percent_portion:]
    
    self.num_of_label_tumor_trn=len(label_tumor_trn)
    self.num_of_label_tumor_vali=len(label_tumor_vali)
    # print("self.num_of_label_tumor_trn",self.num_of_label_tumor_trn)
    # print("self.num_of_label_tumor_vali",self.num_of_label_tumor_vali)
    # 198022
    # 22003

    # ================================================================================
    # @ Create pairs of "train image path" and "label"

    self.tumor_trn_pairs=list(zip(path_trn,label_tumor_trn))
    shuffle(self.tumor_trn_pairs)

    # ================================================================================
    # @ Create pairs of validation image path and label

    self.tumor_vali_pairs=list(zip(path_vali,label_tumor_vali))

    # ================================================================================
    # @ Perform screening on entire sorted pairs of "image path" and "id in label",
    # to see whether thye are matched

    label_tumor_sorted_np=np.array(label_tumor_sorted_list)

    # --------------------------------------------------------------------------------
    # 1. Check number of images on both cases
    # print("num_imgs",num_imgs)
    # 220025

    num_ids=label_tumor_sorted_np[:,0].shape[0]
    # print("num_ids",num_ids)
    # 220025

    assert num_imgs==num_ids,'The number of paths in text file and the number of ids in cvs file are different'

    # --------------------------------------------------------------------------------
    # 2. Check pair of images

    zipped_pairs=zip(path_of_imgs_tumor,label_tumor_sorted_np[:,0])

    for one_pair in list(zipped_pairs):
      # print("one_pair",one_pair)
      # ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/00/00001b2b5609af42ab0ab276dd4cd41c3e7745b5.tif\n', 
      #  '00001b2b5609af42ab0ab276dd4cd41c3e7745b5')

      one_path_in_text=one_pair[0].replace("\n","").split("/")[-1].split(".")[0]
      one_id_in_label=one_pair[1]
      # print("one_path_in_text",one_path_in_text)
      # print("one_id_in_label",one_id_in_label)
      # 00001b2b5609af42ab0ab276dd4cd41c3e7745b5
      # 00001b2b5609af42ab0ab276dd4cd41c3e7745b5

      assert one_path_in_text==one_id_in_label,\
        'Different pair detected: \n'+\
        'in text file: '+str(one_path_in_text)+"\n"+\
        'in csv file: '+str(one_id_in_label)
  
  def __len__(self):
    if self.is_train==True:
      # Return number of entire train images
      return self.num_of_label_tumor_trn

    else:
      # Return number of entire test images
      return self.num_of_label_tumor_vali

  def __getitem__(self,idx):
    if self.is_train==True:
      # Return one pair of train image and label
      one_pair=self.tumor_trn_pairs[idx]
      return one_pair

    else:
      # Return one pair of test image and label
      one_pair=self.tumor_vali_pairs[idx]
      return one_pair

  # def __call__(self):
  #   return 1
