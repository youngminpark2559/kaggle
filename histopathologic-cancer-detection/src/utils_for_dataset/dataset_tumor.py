
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
from src.utils import utils_data_for_dataset_class as utils_data_for_dataset_class

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
    # c list_containing_paths: list which contains paths of images
    list_containing_paths,num_imgs=\
      utils_common.return_path_list_from_txt(txt_containing_paths)
    # print("self.list_containing_paths,",self.list_containing_paths)
    # ['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/00001b2b5609af42ab0ab276dd4cd41c3e7745b5.tif\n',
    #  ...
    # print("self.num_imgs",self.num_imgs)
    # 220025
    # c ninety_percent_portion: 90% from entire images
    ninety_percent_portion=int(num_imgs*0.9)
    # print("ninety_percent_portion",ninety_percent_portion)
    # 198022
    # c path_trn: divided paths (90%) for train
    path_trn=list_containing_paths[:ninety_percent_portion]
    # c path_test: divided paths (10%) for test
    path_test=list_containing_paths[ninety_percent_portion:]
    # c num_imgs_of_path_trn: number of train images
    num_imgs_of_path_trn=len(path_trn)
    # c num_imgs_of_path_test: number of test images
    num_imgs_of_path_test=len(path_test)
    # print("self.num_imgs_of_path_trn",self.num_imgs_of_path_trn)
    # print("self.num_imgs_of_path_test",self.num_imgs_of_path_test)
    # 198022
    # 22003
    # c label_tumor: label to images which is loaded from CSV file
    label_tumor=pd.read_csv(txt_containing_labels,encoding='utf8')
    # c label_tumor_sorted: sorted one by ascending ID
    label_tumor_sorted=label_tumor.sort_values(by=["id"], ascending=True)
    # print("label_tumor_sorted",label_tumor_sorted.head())
    #                                               id  label
    # 151577  00001b2b5609af42ab0ab276dd4cd41c3e7745b5      1
    # 16166   000020de2aa6193f4c160e398a8edea95b1da598      0
    # 87832   00004aab08381d25d315384d646f5ce413ea24b1      0
    # 69359   0000d563d5cfafc4e68acb7c9829258a298d9b6a      0
    # 128953  0000da768d06b879e5754c43e2298ce48726f722      1
    # c label_tumor_sorted_list: label in list
    label_tumor_sorted_list=label_tumor_sorted.iloc[:,:].values.tolist()
    # print("label_tumor_sorted_list",label_tumor_sorted_list)
    # [['00001b2b5609af42ab0ab276dd4cd41c3e7745b5', 1], 
    #  ['000020de2aa6193f4c160e398a8edea95b1da598', 0],
    #  ...
    # c label_tumor_trn: labels for train images
    label_tumor_trn=label_tumor_sorted_list[:ninety_percent_portion]
    # c label_tumor_test: labels for test images
    label_tumor_test=label_tumor_sorted_list[ninety_percent_portion:]
    self.num_of_label_tumor_trn=len(label_tumor_trn)
    self.num_of_label_tumor_test=len(label_tumor_test)
    # print("num_of_label_tumor_trn",num_of_label_tumor_trn)
    # print("num_of_label_tumor_test",num_of_label_tumor_test)
    # 198022
    # 22003
    self.tumor_trn_pairs=list(zip(path_trn,label_tumor_trn))
    shuffle(self.tumor_trn_pairs)
    # print("self.tumor_trn_pairs",self.tumor_trn_pairs)
    # Before shuffe
    # [('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/00001b2b5609af42ab0ab276dd4cd41c3e7745b5.tif\n',
    #  ['00001b2b5609af42ab0ab276dd4cd41c3e7745b5',1]),
    #  ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/000020de2aa6193f4c160e398a8edea95b1da598.tif\n',
    #  ['000020de2aa6193f4c160e398a8edea95b1da598',0]),
    # After shuffle
    # [('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/0f900cbdb478981850b4f3e4b35f09f339e5688d.tif\n',
    #  ['0f900cbdb478981850b4f3e4b35f09f339e5688d',0]),
    #  ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/3eec3fd2341805717e5adacd2fc0089f68410977.tif\n',
    #  ['3eec3fd2341805717e5adacd2fc0089f68410977',0]),
    self.tumor_test_pairs=list(zip(path_test,label_tumor_test))
    # @ Perform screening on entire sorted pairs of "image path" and "id in label" are matched
    label_tumor_sorted_np=np.array(label_tumor_sorted_list)
    # 1. Check number of images on both cases
    # print("self.num_imgs",self.num_imgs)
    # 220025
    num_ids=label_tumor_sorted_np[:,0].shape[0]
    # print("num_ids",num_ids)
    # 220025
    assert num_imgs==num_ids,'The number of paths in text file and the number of ids in cvs file are different'
    # 2. Check pair of images
    zipped_pairs=zip(list_containing_paths,label_tumor_sorted_np[:,0])
    for one_pair in list(zipped_pairs):
      # print("one_pair",one_pair)
      # ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/00001b2b5609af42ab0ab276dd4cd41c3e7745b5.tif\n', 
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
      return self.num_of_label_tumor_test

  def __getitem__(self,idx):
    if self.is_train==True:
      # Return one pair of train image and label
      one_pair=self.tumor_trn_pairs[idx]
      return one_pair
    else:
      # Return one pair of test image and label
      one_pair=self.tumor_test_pairs[idx]
      return one_pair

  # def __call__(self):
  #   return 1
