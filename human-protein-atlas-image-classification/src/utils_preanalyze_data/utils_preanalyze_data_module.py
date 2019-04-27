import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth',-1);pd.set_option('display.max_columns',None)
import argparse
import os
import copy
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

# ================================================================================
from src.utils import utils_image as utils_image
from src.utils import utils_common as utils_common

# ================================================================================
train_imgs="/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/Path_of_train_images.txt"
label_csv="/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train.csv"

# ================================================================================
label_names={
  0:"Nucleoplasm",  
  1:"Nuclear membrane",   
  2:"Nucleoli",   
  3:"Nucleoli fibrillar center",   
  4:"Nuclear speckles",
  5:"Nuclear bodies",   
  6:"Endoplasmic reticulum",   
  7:"Golgi apparatus",   
  8:"Peroxisomes",   
  9:"Endosomes",   
  10:"Lysosomes",   
  11:"Intermediate filaments",   
  12:"Actin filaments",   
  13:"Focal adhesion sites",   
  14:"Microtubules",   
  15:"Microtubule ends",   
  16:"Cytokinetic bridge",   
  17:"Mitotic spindle",   
  18:"Microtubule organizing center",   
  19:"Centrosome",   
  20:"Lipid droplets",   
  21:"Plasma membrane",   
  22:"Cell junctions",   
  23:"Mitochondria",   
  24:"Aggresome",   
  25:"Cytosol",   
  26:"Cytoplasmic bodies",   
  27:"Rods & rings"}

num_trn_lbls=len(list(label_names.keys()))
# 28

# ================================================================================
def visualize_images(args):
 
  # ================================================================================
  loaded_path,num_imgs=utils_common.return_path_list_from_txt(train_imgs)
  # print("loaded_path",loaded_path)
  # ['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_blue.png\n', 
  #  '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_green.png\n',
  
  # print("num_imgs",num_imgs)
  # 124288

  loaded_path_chunked=[]
  for i in range(0,int(num_imgs/4),4):
    one_protein=loaded_path[i:i+4]
    # print("one_protein",one_protein)
    # ['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_blue.png\n',
    #  '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_green.png\n',
    #  '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_red.png\n',
    #  '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_yellow.png\n']

    loaded_path_chunked.append(one_protein)
  
  # print("loaded_path_chunked",loaded_path_chunked)
  # [['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_blue.png\n',
  #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_green.png\n',
  #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_red.png\n',
  #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_yellow.png\n'

  path_3_proteins=loaded_path_chunked[:3]

  # ================================================================================
  images_of_3_proteins=[]
  for one_protein in path_3_proteins:
    b_img=one_protein[0].replace("\n","")
    g_img=one_protein[1].replace("\n","")
    r_img=one_protein[2].replace("\n","")
    y_img=one_protein[3].replace("\n","")

    b_img=utils_image.load_img(b_img)
    g_img=utils_image.load_img(g_img)
    r_img=utils_image.load_img(r_img)
    y_img=utils_image.load_img(y_img)

    # print("b_img",b_img.shape)
    # (512, 512)
    # print("g_img",g_img.shape)
    # (512, 512)
    # print("r_img",r_img.shape)
    # (512, 512)
    # print("y_img",y_img.shape)
    # (512, 512)
    
    images_of_3_proteins.append([b_img,g_img,r_img,y_img])

  i=0
  for one_protein_img in images_of_3_proteins:
    bg_img=np.zeros((one_protein_img[0].shape[0],one_protein_img[0].shape[1],3))
    # print("bg_img",bg_img.shape)
    # (512, 512, 3)

    bg_img_flat_for_b=bg_img.reshape(-1,3).copy()
    bg_img_flat_for_g=bg_img.reshape(-1,3).copy()
    bg_img_flat_for_r=bg_img.reshape(-1,3).copy()
    bg_img_flat_for_y=bg_img.reshape(-1,3).copy()


    # print("one_protein_img[0]",one_protein_img[0].shape)
    # (512, 512)
    # print("one_protein_img[1]",one_protein_img[1].shape)
    # (512, 512)
    # print("one_protein_img[2]",one_protein_img[2].shape)
    # (512, 512)
    # print("one_protein_img[3]",one_protein_img[3].shape)
    # (512, 512)

    b_img=one_protein_img[0].reshape(-1)
    g_img=one_protein_img[1].reshape(-1)
    r_img=one_protein_img[2].reshape(-1)
    y_img=one_protein_img[3].reshape(-1)
    
    rgb_img=np.stack((one_protein_img[0],one_protein_img[1],one_protein_img[2])).transpose(1,2,0)
    # print("rgb_img",rgb_img.shape)

    # ================================================================================
    import scipy.misc
    scipy.misc.imsave('./img_out/rgb_img_'+str(i)+'.png',rgb_img)

    bg_img_flat_for_b[:,2]=b_img
    scipy.misc.imsave('./img_out/b_img_'+str(i)+'.png',bg_img_flat_for_b.reshape(512,512,3))

    bg_img_flat_for_g[:,1]=g_img
    scipy.misc.imsave('./img_out/g_img_'+str(i)+'.png',bg_img_flat_for_g.reshape(512,512,3))

    bg_img_flat_for_r[:,0]=r_img
    scipy.misc.imsave('./img_out/r_img_'+str(i)+'.png',bg_img_flat_for_r.reshape(512,512,3))

    bg_img_flat_for_y[:,0]=y_img
    scipy.misc.imsave('./img_out/y_img_'+str(i)+'.png',bg_img_flat_for_y.reshape(512,512,3))
        
    i=i+1

# ================================================================================
def load_train_label_csv(args):

  train_label_csv=pd.read_csv(label_csv,encoding='utf8')
  # print("train_label_csv",train_label_csv)
  #                                          Id   Target
  # 0      00070df0-bbc3-11e8-b2bc-ac1f6b6435d0     16 0
  # 1      000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0  7 1 2 0
  # 2      000a9596-bbc4-11e8-b2bc-ac1f6b6435d0        5

  # ================================================================================
  train_label_csv_sorted=train_label_csv.sort_values(by=["Id"],ascending=True)
  # print("train_label_csv_sorted",train_label_csv_sorted)
  #                                          Id   Target
  # 0      00070df0-bbc3-11e8-b2bc-ac1f6b6435d0     16 0
  # 1      000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0  7 1 2 0

  # ================================================================================
  # print(train_label_csv_sorted.shape)
  # (31072, 2)

  num_trn_imgs=train_label_csv_sorted.shape[0]

  # ================================================================================
  trn_multi_lbl_in_one_hot=np.zeros((num_trn_imgs,num_trn_lbls))
  # print("trn_multi_lbl_in_one_hot",trn_multi_lbl_in_one_hot.shape)
  # (31072, 28)

  # ================================================================================
  # print("np.array(train_label_csv_sorted)",np.array(train_label_csv_sorted))
  # [['00070df0-bbc3-11e8-b2bc-ac1f6b6435d0' '16 0']
  #  ['000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0' '7 1 2 0']

  multi_lbls_of_each_img=np.array(train_label_csv_sorted)[:,1]
  # print("multi_lbls_of_each_img",multi_lbls_of_each_img)
  # ['16 0' '7 1 2 0' '5' ... '7' '25 2 21' '2 0']

  # ================================================================================
  for i,one_multi_lbl in enumerate(multi_lbls_of_each_img):

    one_multi_lbl_li=one_multi_lbl.split(" ")
    # print("one_multi_lbl_li",one_multi_lbl_li)
    # ['16', '0']

    for one_lbl in one_multi_lbl_li:
      # print("one_lbl",one_lbl)
      # 16

      trn_multi_lbl_in_one_hot[i,int(one_lbl)]=1
      # print("trn_multi_lbl_in_one_hot",trn_multi_lbl_in_one_hot)

  # print("trn_multi_lbl_in_one_hot",trn_multi_lbl_in_one_hot)
  # [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  #  [1. 1. 1. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

  # ================================================================================
  trn_multi_lbl_in_one_hot_pd=pd.DataFrame(trn_multi_lbl_in_one_hot)
  # print("trn_multi_lbl_in_one_hot_pd",trn_multi_lbl_in_one_hot_pd)
  #        0    1    2    3    4    5    6  ...    21   22   23   24   25   26   27
  # 0      1.0  0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0
  # 1      1.0  1.0  1.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0

  # ================================================================================
  train_label_csv_onehot=pd.concat([train_label_csv_sorted,trn_multi_lbl_in_one_hot_pd],axis=1)
  # print("concat_trn",concat_trn)
  #                                          Id   Target    0 ...    25   26   27
  # 0      00070df0-bbc3-11e8-b2bc-ac1f6b6435d0     16 0  1.0 ...   0.0  0.0  0.0
  # 1      000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0  7 1 2 0  1.0 ...   0.0  0.0  0.0

  # print(train_label_csv_onehot.shape)
  # (31072, 30)

  # ================================================================================
  for k,v in label_names.items():
    train_label_csv_onehot=train_label_csv_onehot.rename(columns={k:v})
  
  # print("train_label_csv_onehot",train_label_csv_onehot.head(2))
  #                                      Id   Target  Nucleoplasm  \
  # 0  00070df0-bbc3-11e8-b2bc-ac1f6b6435d0  16 0     1.0           
  # 1  000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0  7 1 2 0  1.0           

  #   Nuclear membrane  Nucleoli  Nucleoli fibrillar center  Nuclear speckles  \
  # 0  0.0               0.0       0.0                        0.0                
  # 1  1.0               1.0       0.0                        0.0                

  #   Nuclear bodies  Endoplasmic reticulum  Golgi apparatus  Peroxisomes  \
  # 0  0.0             0.0                    0.0              0.0           
  # 1  0.0             0.0                    1.0              0.0           

  #   Endosomes  Lysosomes  Intermediate filaments  Actin filaments  \
  # 0  0.0        0.0        0.0                     0.0               
  # 1  0.0        0.0        0.0                     0.0               

  #   Focal adhesion sites  Microtubules  Microtubule ends  Cytokinetic bridge  \
  # 0  0.0                   0.0           0.0               1.0                  
  # 1  0.0                   0.0           0.0               0.0                  

  #   Mitotic spindle  Microtubule organizing center  Centrosome  Lipid droplets  \
  # 0  0.0              0.0                            0.0         0.0              
  # 1  0.0              0.0                            0.0         0.0              

  #   Plasma membrane  Cell junctions  Mitochondria  Aggresome  Cytosol  \
  # 0  0.0              0.0             0.0           0.0        0.0       
  # 1  0.0              0.0             0.0           0.0        0.0       

  #   Cytoplasmic bodies  Rods & rings  
  # 0  0.0                 0.0           
  # 1  0.0                 0.0           

  return train_label_csv_onehot

# ================================================================================
def frequent_distribution_of_train_label_data(args):
  train_label_csv_onehot=load_train_label_csv(args)

  # ================================================================================
  # c tl_oh_Id_Target_drped: train label one hot, Id and Target dropped
  tl_oh_Id_Target_drped=trn_labels_dropped=train_label_csv_onehot.drop(["Id","Target"],axis=1)

  trn_labels_summed=tl_oh_Id_Target_drped.sum(axis=0)
  # print("trn_labels_summed",trn_labels_summed)
  # Nucleoplasm                      12885.0
  # Nuclear membrane                 1254.0 

  target_counts=trn_labels_summed.sort_values(ascending=False)
  # print("target_counts",target_counts)
  # Nucleoplasm                      12885.0
  # Cytosol                          8228.0 

  data_of_index=target_counts.index.values
  # print("data_of_index",data_of_index)
  # ['Nucleoplasm' 'Cytosol'

  data_of_values=target_counts.values
  # print("data_of_values",data_of_values)
  # [1.2885e+04 8.2280e+03

  plt.figure(figsize=(15,15))
  sns.barplot(y=data_of_index,x=data_of_values,order=target_counts.index)
  plt.show()
  # plt.savefig("train_label_distribution.png")

# ================================================================================
def frequent_distribution_of_number_of_labels_to_each_img(args):
  train_label_csv_onehot=load_train_label_csv(args)
  train_label_csv_onehot["number_of_targets"] = train_label_csv_onehot.drop(["Id", "Target"],axis=1).sum(axis=1)
  count_perc = np.round(100 * train_label_csv_onehot["number_of_targets"].value_counts() / train_label_csv_onehot.shape[0], 2)
  plt.figure(figsize=(20,5))
  sns.barplot(x=count_perc.index.values, y=count_perc.values, palette="Reds")
  plt.xlabel("Number of targets per image")
  plt.ylabel("% of train data")
  # plt.show()
  plt.savefig("Frequent_distribution_of_number_of_labels_to_each_img.png")
  
  return train_label_csv_onehot
# ================================================================================
def correlation_of_proteins(args):
  # train_label_csv_onehot=load_train_label_csv(args)
  # train_labels=train_label_csv_onehot
  train_label_csv_onehot=frequent_distribution_of_number_of_labels_to_each_img(args)
  train_labels=train_label_csv_onehot

  # ================================================================================
  # Let's see if we find some correlations between our targets. This way we may already see that some proteins often come together.

  plt.figure(figsize=(15,15))

  trn_lbl_num_targets=train_labels.number_of_targets
  # print("trn_lbl_num_targets",trn_lbl_num_targets)
  # 0        2.0
  # 1        4.0
  # 2        1.0

  mask_for_gt_1=train_labels.number_of_targets>1
  # print("mask_for_gt_1",mask_for_gt_1)
  # 0        True 
  # 1        True 
  # 2        False

  masked_trn_lbls=train_labels[mask_for_gt_1]
  # print("masked_trn_lbls",masked_trn_lbls)
  #                                          Id    Target  Nucleoplasm  \
  # 0      00070df0-bbc3-11e8-b2bc-ac1f6b6435d0  16 0      1.0           
  # 1      000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0  7 1 2 0   1.0           

  drop_masked_trn_lbls=masked_trn_lbls.drop(["Id","Target","number_of_targets"],axis=1)
  # print("drop_masked_trn_lbls",drop_masked_trn_lbls)
  #        Nucleoplasm  Nuclear membrane  Nucleoli  Nucleoli fibrillar center  \
  # 0      1.0          0.0               0.0       0.0                         
  # 1      1.0          1.0               1.0       0.0                         

  corr_of_masked_trn_lbls=drop_masked_trn_lbls.corr()
  # print("corr_of_masked_trn_lbls",corr_of_masked_trn_lbls)
  #                                Nucleoplasm  Nuclear membrane  Nucleoli  \
  # Nucleoplasm                    1.000000    -0.024980         -0.038865   
  # Nuclear membrane              -0.024980     1.000000         -0.018065   

  sns.heatmap(corr_of_masked_trn_lbls,cmap="RdYlBu",vmin=-1,vmax=1)
  plt.show()
  # plt.savefig("correlation_of_proteins.png")


# correlation of seldom proteins Lysosomes and endosomes
# # How are special and seldom targets grouped?

# # Lysosomes and endosomes
# # Let's start with these high correlated features!

# def find_counts(special_target, labels):
#     counts = labels[labels[special_target] == 1].drop(
#         ["Id", "Target", "number_of_targets"],axis=1
#     ).sum(axis=0)
#     counts = counts[counts > 0]
#     counts = counts.sort_values()
#     return counts

# lyso_endo_counts = find_counts("Lysosomes", train_labels)

# plt.figure(figsize=(10,3))
# sns.barplot(x=lyso_endo_counts.index.values, y=lyso_endo_counts.values, palette="Blues")
# plt.ylabel("Counts in train data")
# # plt.show()  