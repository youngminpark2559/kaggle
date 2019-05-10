
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

class custom_ds_Submission(data.Dataset):
  def __init__(self):
    
    label_submission=pd.read_csv("/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/sample_submission.csv",encoding='utf8')
    label_submission_list=label_submission.iloc[:,:].values.tolist()
    # print("label_submission_list",label_submission_list)
    # [['0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5', 0], 

    self.image_paths=[]

    for one_pair in label_submission_list:
      one_img_name=one_pair[0]
      one_label=one_pair[1]
      # print("one_img_name",one_img_name)
      # print("one_label",one_label)
      # 0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5
      # 0

      file_path="/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/test"+"/"+one_img_name+".tif"
      # print("file_path",file_path)
      # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/test/0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5.tif

      self.image_paths.append(file_path)

    
    self.num_imgs=len(self.image_paths)
  
  def __len__(self):
    return self.num_imgs

  def __getitem__(self,idx):
    one_img_path=self.image_paths[idx]
    return one_img_path

  # def __call__(self):
  #   return 1
