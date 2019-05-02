
# /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/prj_root/src/utils_for_dataset/dataset_cgmit.py
# /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/prj_root/src/unit_test/Test_dataset_cgmit.py

# ================================================================================
import csv
import numpy as np
import pandas as pd
from random import shuffle

# ================================================================================
import torch.utils.data as data

# ================================================================================
from src.utils import utils_common as utils_common
from src.utils import utils_image as utils_image

# ================================================================================
class Custom_DS(data.Dataset):
  def __init__(self,single_train_k,single_train_lbl_k,args):
    # print("single_train_k",single_train_k)
    # [['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_blue.png\n'
    #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_green.png\n'
    #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_red.png\n'
    #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_yellow.png\n']

    # print("single_train_lbl_k",single_train_lbl_k)
    # [['000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0' '7 1 2 0']
    #  ['000a9596-bbc4-11e8-b2bc-ac1f6b6435d0' '5']
    #  ['000c99ba-bba4-11e8-b2b9-ac1f6b6435d0' '1']
    #  ...
    #  ['ffe55eba-bbba-11e8-b2ba-ac1f6b6435d0' '5 0']
    #  ['ffe61798-bbc3-11e8-b2bc-ac1f6b6435d0' '19 23']
    #  ['ffe8cf0c-bba9-11e8-b2ba-ac1f6b6435d0' '18']]

    # print("single_train_lbl_k",single_train_lbl_k[:,1])
    # ['7 1 2 0' '5' '1' ... '5 0' '19 23' '18']

    # set_of_labels=[list(map(int,one_label_set.split(" "))) for one_label_set in single_train_lbl_k[:,1]]
    # print("set_of_labels",set_of_labels)
    # [[7, 1, 2, 0],

    zipped=np.array(list(zip(single_train_k,single_train_lbl_k[:,1])))
    # print("zipped",zipped)
    # [[array(['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_blue.png',
    #          '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_green.png',
    #          '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_red.png',
    #          '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_yellow.png'],dtype='<U140')
    #  '7 1 2 0']

    # ================================================================================
    zipped_new=[[list(one_protein_pair[0]),one_protein_pair[1]] for one_protein_pair in zipped]
    # print("zipped_new",zipped_new)
    # [[['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_blue.png',
    #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_green.png',
    #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_red.png',
    #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_yellow.png'],
    #  '7 1 2 0'],
 
    # took_time_min 0:00:00.130517

    shuffle(zipped_new)
    # print("zipped_new",zipped_new)
    # [[['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/e6e2c6c0-bbae-11e8-b2ba-ac1f6b6435d0_blue.png',
    #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/e6e2c6c0-bbae-11e8-b2ba-ac1f6b6435d0_green.png',
    #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/e6e2c6c0-bbae-11e8-b2ba-ac1f6b6435d0_red.png',
    #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/e6e2c6c0-bbae-11e8-b2ba-ac1f6b6435d0_yellow.png'],
    #  '3 0'],

    self.trn_pairs=zipped_new

    # ================================================================================
    # instance of argument 
    self.args=args

    # ================================================================================
    self.nb_trn_imgs=len(single_train_k)
    # 20714

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
    # [['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_blue.png'
    #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_green.png'

    # print("single_vali_lbl_k",single_vali_lbl_k)
    # [['00070df0-bbc3-11e8-b2bc-ac1f6b6435d0' '16 0']
    #  ['001838f8-bbca-11e8-b2bc-ac1f6b6435d0' '18']

    # print("single_vali_lbl_k",single_vali_lbl_k[:,1])
    # ['7 1 2 0' '5' '1' ... '5 0' '19 23' '18']

    zipped=np.array(list(zip(single_vali_k,single_vali_lbl_k[:,1])))

    # ================================================================================
    zipped_new=[[list(one_protein_pair[0]),one_protein_pair[1]] for one_protein_pair in zipped]
    # print("zipped_new",zipped_new)
    # [[['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_blue.png',
    #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_green.png',
    #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_red.png',
    #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_yellow.png'],
    #  '16 0'],

    self.vali_pairs=zipped_new
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
