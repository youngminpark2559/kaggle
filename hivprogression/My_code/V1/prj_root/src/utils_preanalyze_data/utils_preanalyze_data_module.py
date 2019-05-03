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
train_data="/mnt/1T-5e7/mycodehtml/prac_data_science/kaggle/hivprogression/My_code/Data/training_data.csv"
test_data="/mnt/1T-5e7/mycodehtml/prac_data_science/kaggle/hivprogression/My_code/Data/test_data.csv"
# Index(['PatientID', Resp', 'PR Seq', 'RT Seq', 'VL-t0', 'CD4-t0'], dtype='object')

# ================================================================================
def load_csv_file(path):
  loaded_csv=pd.read_csv(path,encoding='utf8')
  return loaded_csv

# ================================================================================
def check_nan(df):
  ret=df.isna()
  print("ret",ret)
  #      PatientID   Resp  PR Seq  RT Seq  VL-t0  CD4-t0
  # 0    False      False  False   False   False  False 
  # 1    False      False  False   False   False  False 

  # ================================================================================
  sum_nan=ret.sum()
  print("sum_nan",sum_nan)
  # PatientID    0 
  # Resp         0 
  # PR Seq       80
  # RT Seq       0 
  # VL-t0        0 
  # CD4-t0       0 

def process_data_until_being_useful(train_data,args):
  train_csv=load_csv_file(train_data)
  # print("train_csv",train_csv.head(2))
  #    PatientID  Resp  \
  # 0  1          0      
  # 1  2          0      

  #                                                                                                                                                                                                                                                                                                       PR Seq  \
  # 0  CCTCAAATCACTCTTTGGCAACGACCCCTCGTCCCAATAAGGATAGGGGGGCAACTAAAGGAAGCYCTATTAGATACAGGAGCAGATGATACAGTATTAGAAGACATGGAGTTGCCAGGAAGATGGAAACCAAAAATGATAGGGGGAATTGGAGGTTTTATCAAAGTAARACAGTATGATCAGRTACCCATAGAAATCTATGGACATAAAGCTGTAGGTACAGTATTAATAGGACCTACACCTGTCAACATAATTGGAAGAAATCTGTTGACTCAGCTTGGTTGCACTTTAAATTTY   
  # 1  CCTCAAATCACTCTTTGGCAACGACCCCTCGTCGCAATAAAGATAGGGGGGCAACTAAAGGAAGCTCTATTAGATACAGGAGCAGATGATACAGTATTAGAAGACATGGAATTGCCAGGAAGATGGAAACCAAAAATAATAGGGGGAATTGGAGGTTTTATCAAAGTAAGACAGTATGATCAGATACCCATAGAAATCTGTGGACATAAAGTTATAAGTACAGTATTAATAGGACCTACACCTGTCAACATAATTGGAAGAAATCTGATGACTCAGCTTGGTTGCACTTTAAATTTT   

  #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           RT Seq  \
  # 0  CCCATTAGTCCTATTGAAACTGTACCAGTAAAGCTAAAGCCAGGAATGGATGGCCCAAAAGTTAAACAATGGCCATTGACAGAAGAAAAAATAAAAGCATTAGTAGAAATTTGYACAGAAATGGAAAAGGAAGGGAAAATTTCAAAAATTGGGCCTGAAAATCCATATAATACTCCAGTATTTGCCATAAAGAAAAAAGACAGTACTACATGGAGAAAATTAGTAGATTTCAGAGAACTTAATAAGAGAACTCAAGACTTCTGGGAAGTTCAAYTAGGAATACCACATCCCGCWGGGTTAAAAAAGAAYAAATCAGTAACAGTACTGGATGTGGGTGATGCATATTTCTCAGTTCCMTTAGATAAAGACTTCAGGAAGTATACTGCATTTACCATACCTAGTATAAACAATGAGACACCAGGGATTAGATATCAGTACAATGTGCTTCCACAGGGATGGAAAGGATCACCAGCAATATTCCAAAGTAGCATGACAAAAATCTTAGAGCCTTTTAGAAAACGAAATCCAGACATAGTTATCTACCAATACATGGATGATTTGTATGTAGGATCTGATTTRGAAATAGAACAGCATAGAACAAAAATAGAGGAACTGAGACAACATCTGTCAAGGTGGGGGTTTACCACACCAGACAAAAAACATCAGAAAGAACCTCCATTCCTTTGGATGGGCTATGAACTCCATCCTGATAAATGGACAGTACAGCCTATAGTTCTGCCAGAAAAAGATAGCTGGACTGTCAATGACATACAGAAGTTAGTGGGGAAGTTGAATTGGGCAAGTCAGATTTAYGCAGGGATTAAAGTAAAGCAATTATGTAAACTCCTTAGGGGGACCAAGKCACTAACAGAAATAATACCACTAACAAGAGAAGCAGAGCTAGAACTGGCAGAAAACAGGGAAATTCTAAAAGAACCAGTACATGGAGTGTATTATGATCCAACAAAAGACTTAATAGCAGAAATACAGAAGCAGGGGCAAGGC   
  # 1  CCCATTAGTCCTATTGAAACTGTACCAGTAAAATTAAAGCCAGGAATGGATGGCCCAAAAGTTAAACAATGGCCATTGACAGAAGAAAAAATAAAAGCATTAGTAGAAATTTGTACAGARATGGAAARGGARGGGAAAATTTCAAAAATTGGGCCTGAAAATCCATACAATACTCCGGTATTTGTCATAAAGAAAAAGGACAGTACTAAGTGGAGAAAAGTAGTAGATTTCAGAGAACTTAATAAAAGAACTCAAGACTTCTGGGAAGTTCAATTAGGGATACCACATCCCGCAGGGWTAAAAAAGAATAAATCAGTAACAGTATTGGATGTGGGTGATGCATACTTTTCAGTTCCCTTAGATGAAGACTTCAGGAAGTATACTGCATTTACCATACCCAGTACAAACAATGAGACACCAGGGATTAGATATCAGTACAATGTGCTTCCACAGGGATGGAAAGGATCACCAGCAATATTCCAAAGTAGCATGACAAAAATTTTAGAGCCTTTTAGAAAACAAAATCCAGACATAGTTATCTATCAATACGTGGATGATTTGTATGTAGGATCTGACTTAGAAATAGGGCAGCATAGAACAAAAATAGAGGAACTGAGACAACATCTGCTGAAGTGGGGATTGACCACACCAGACAAAAAAYATCAGAAAGAACCTCCATTTCGTTGGATGGGTTATGAACTCCATCCTGATAAMTGGACAGTACAGCCTATAGTGCTGCCAGAAAAAGACAGCTGGACTGTCAATGACATACAGAAGTTAGTGGGAAAATTAAATTGGGCAAGCCAGATTTACGCAGGGATTAAAGTAAAGCAATTATGTAAACTCCTTAGGGGAACCAAGGCACTAACWGATGTAATACCACTAACAAGAGAAGCAGAGCTAGAACTG                                                                                                   

  #   VL-t0  CD4-t0  
  # 0  4.3    145     
  # 1  3.6    224     

  # ================================================================================
  # print("train_csv",train_csv.shape)
  # (1000, 6)

  # ================================================================================
  # check_nan(train_csv)
  
  # ================================================================================
  train_csv=train_csv.dropna()

  return train_csv

# ================================================================================
def see_correlation_of_data(args):

  train_csv=process_data_until_being_useful(train_data,args)
  # print("train_csv",train_csv.shape)
  # (920, 6)

  # ================================================================================
  train_csv_wo_id=train_csv.iloc[:,1:]
  # print("train_csv_wo_id",train_csv_wo_id.shape)
  # (920, 5)

  # ================================================================================
  cor_mat=train_csv_wo_id.corr()
  # print("cor_mat",cor_mat.shape)
  # (3, 3)

  # print("cor_mat",cor_mat)
  #             Resp     VL-t0    CD4-t0
  # Resp    1.000000  0.363947 -0.127548
  # VL-t0   0.363947  1.000000 -0.427281
  # CD4-t0 -0.127548 -0.427281  1.000000

  # ================================================================================
  # * Normalize data
  # * https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range

  cor_mat_np=np.array(cor_mat,dtype="float16")
  cor_mat_np_sh=cor_mat_np.shape

  min_in_arr=np.min(cor_mat_np.reshape(-1))
  max_in_arr=np.max(cor_mat_np.reshape(-1))
  # print("min_in_arr",min_in_arr)
  # print("max_in_arr",max_in_arr)
  # -0.4272
  # 1.0

  # c norm_corr_mat_np: normalized correlation matrix in np
  norm_corr_mat_np=(cor_mat_np-min_in_arr)/(max_in_arr-min_in_arr)
  # print("norm_corr_mat_np",norm_corr_mat_np)
  # [[1.    0.554 0.21 ]
  #  [0.554 1.    0.   ]
  #  [0.21  0.    1.   ]]

  norm_corr_mat_df=pd.DataFrame(norm_corr_mat_np)
  # print("norm_corr_mat_df",norm_corr_mat_df)
  #           0         1         2
  # 0  1.000000  0.554199  0.209961
  # 1  0.554199  1.000000  0.000000
  # 2  0.209961  0.000000  1.000000

  new_col_name={0:"Resp",1:"VL-t0",2:"CD4-t0"}
  new_idx_name={0:"Resp",1:"VL-t0",2:"CD4-t0"}
  norm_corr_mat_df=norm_corr_mat_df.rename(columns=new_col_name,index=new_idx_name,inplace=False)
  # print("norm_corr_mat_df_a",norm_corr_mat_df_a)
  #             Resp     VL-t0    CD4-t0
  # Resp    1.000000  0.554199  0.209961
  # VL-t0   0.554199  1.000000  0.000000
  # CD4-t0  0.209961  0.000000  1.000000


  sns.heatmap(norm_corr_mat_df)
  plt.show()
  
  # ================================================================================
  # * Meaning
  # * Therea are negative and positive relationships between factors

# ================================================================================
def distribution_of_label_in_trn_data(args):
  train_csv=process_data_until_being_useful(train_data,args)
  train_csv_wo_id=train_csv.iloc[:,1:]

  # ================================================================================
  cols=train_csv_wo_id.columns
  # print("cols",cols)
  # Index(['Resp', 'PR Seq', 'RT Seq', 'VL-t0', 'CD4-t0'], dtype='object')

  # ================================================================================
  resp_data=train_csv_wo_id.iloc[:,0]
  # print("resp_data",resp_data.shape)
  # (920,)

  # ================================================================================
  resp_data_np=np.array(resp_data)
  uniq_resp_data=np.unique(resp_data_np)
  # print("uniq_resp_data",uniq_resp_data)
  # [0 1]

  resp_data_0=resp_data_np[resp_data_np==uniq_resp_data[0]]
  resp_data_1=resp_data_np[resp_data_np==uniq_resp_data[1]]
  # print("resp_data_0",resp_data_0)
  # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  # print("resp_data_1",resp_data_1)
  # [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
  
  # ================================================================================
  num_0=len(resp_data_0)
  num_1=len(resp_data_1)
  # print("num_0",num_0)
  # print("num_1",num_1)
  # 733
  # 187

  # ================================================================================
  objects=('Label 0','Label 1')
  y_pos=np.arange(len(objects))
  num_data=[num_0,num_1]

  plt.bar(y_pos,num_data,align="center",alpha=0.5)
  plt.xticks(y_pos,objects)
  plt.ylabel('Label')
  plt.title('Distribution of label data (Resp)')
  plt.show()

def distribution_of_PR_Seq_in_trn_data(args):
  train_csv=process_data_until_being_useful(train_data,args)
  # print("train_csv",train_csv.shape)
  # (920, 6)

  # print("train_csv",train_csv.columns)
  # Index(['PatientID', 'Resp', 'PR Seq', 'RT Seq', 'VL-t0', 'CD4-t0'], dtype='object')

  # ================================================================================
  PR_Seq_np=np.array(train_csv.iloc[:,2])
  
  # ================================================================================
  uniq_PR_Seq=np.unique(PR_Seq_np)
  # print("uniq_PR_Seq",uniq_PR_Seq)
  # ['CCTCAAATCACTCTTTGGCAACGACCCCTCGTCCCAATAAGGATAGGGGGGCAACTAAAGGAAGCYCTATTAGATACAGGAGCAGATGATACAGTATTAGAAGACATGGAGTTGCCAGGAAGATGGAAACCAAAAATGATAGGGGGAATTGGAGGTTTTATCAAAGTAARACAGTATGATCAGRTACCCATAGAAATCTATGGACATAAAGCTGTAGGTACAGTATTAATAGGACCTACACCTGTCAACATAATTGGAAGAAATCTGTTGACTCAGCTTGGTTGCACTTTAAATTTY'
  
  # ================================================================================
  num_elements=[]
  for one_val in uniq_PR_Seq:
    mask_arr=PR_Seq_np==one_val
    # print("mask_arr",mask_arr.shape)
    # (920,)

    summed_val=np.sum(mask_arr)
    # print("summed_val",summed_val)
    # 2

    num_elements.append(summed_val)

  # print("num_elements",num_elements)
  # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

  # ================================================================================
  # objects=('Label 0','Label 1')
  y_pos=list(range(len(uniq_PR_Seq)))
  num_data=num_elements
  
  plt.plot(y_pos,num_data)
  # plt.xticks(y_pos,objects)
  # plt.ylabel('Label')
  plt.title('Distribution of train data (PR Seq)')
  plt.show()

  # * Meaning:
  # * All data is unique

# ================================================================================
def distribution_of_VL_t0_in_trn_data(args):
  train_csv=process_data_until_being_useful(train_data,args)
  # print("train_csv",train_csv.columns)
  # Index(['PatientID', 'Resp', 'PR Seq', 'RT Seq', 'VL-t0', 'CD4-t0'], dtype='object')

  # ================================================================================
  VL_t0_np=np.array(train_csv.iloc[:,4])

  uniq_VL_t0=np.unique(VL_t0_np)
  # print("uniq_VL_t0",uniq_VL_t0)
  # [2.8  2.9  2.95 3.   3.1  3.2  3.25 3.3  3.35 3.4  3.45 3.5  3.55 3.6
  #  3.65 3.7  3.8  3.85 3.9  4.   4.1  4.15 4.2  4.25 4.3  4.35 4.4  4.5
  #  4.55 4.6  4.7  4.75 4.8  4.85 4.9  5.   5.1  5.2  5.3  5.4  5.45 5.5
  #  5.6  5.7  5.8  5.9  6.  ]

  # ================================================================================
  num_elements=[]
  for one_val in uniq_VL_t0:
    mask_arr=VL_t0_np==one_val
    # print("mask_arr",mask_arr.shape)
    # (920,)

    summed_val=np.sum(mask_arr)
    # print("summed_val",summed_val)
    # 1

    num_elements.append(summed_val)

  # print("num_elements",num_elements)
  # [2, 5, 1, 6, 8, 21, 1, 25, 1, 35, 1, 39, 2, 38, 1, 44, 37, 1, 44, 49, 37, 2, 44, 1, 51, 2, 41, 39, 1, 45, 45, 1, 32, 1, 33, 21, 28, 23, 31, 18, 1, 17, 14, 12, 14, 4, 1]
  
  num_uniq_element=len(num_elements)
  # print("num_uniq_element",num_uniq_element)
  # 920

  # ================================================================================
  # objects=('Label 0','Label 1')
  y_pos=uniq_VL_t0
  num_data=num_elements
  
  plt.plot(y_pos,num_data)
  # plt.xticks(y_pos,objects)
  # plt.ylabel('Label')
  plt.title('Distribution of train data (VL-t0)')
  plt.show()
  
  # * Meaning:
  # * Overall, frequent distribution looks Gaussian normal distribution
  # except for periodic low values like 1 and 2

# ================================================================================
def distribution_of_CD4_t0_in_trn_data(args):
  
  train_csv=process_data_until_being_useful(train_data,args)
  # print("train_csv",train_csv.columns)
  # Index(['PatientID', 'Resp', 'PR Seq', 'RT Seq', 'VL-t0', 'CD4-t0'], dtype='object')

  # ================================================================================
  CD_t0_np=np.array(train_csv.iloc[:,5])

  uniq_CD_t0_np=np.unique(CD_t0_np)
  # print("uniq_CD_t0_np",uniq_CD_t0_np)
  # [   0    1    3    4    5    7    8   10   11   12   13   14   15   16

  # ================================================================================
  num_elements=[]
  for one_val in uniq_CD_t0_np:
    mask_arr=CD_t0_np==one_val
    # print("mask_arr",mask_arr.shape)
    # (920,)

    summed_val=np.sum(mask_arr)
    # print("summed_val",summed_val)
    # 1

    num_elements.append(summed_val)

  # print("num_elements",num_elements)
  # [2, 5, 1, 6, 8, 21, 1, 25, 1, 35, 1, 39, 2, 38, 1, 44, 37, 1, 44, 49, 37, 2, 44, 1, 51, 2, 41, 39, 1, 45, 45, 1, 32, 1, 33, 21, 28, 23, 31, 18, 1, 17, 14, 12, 14, 4, 1]
  
  num_uniq_element=len(num_elements)
  # print("num_uniq_element",num_uniq_element)
  # 920

  # ================================================================================
  # objects=('Label 0','Label 1')
  y_pos=uniq_CD_t0_np
  num_data=num_elements
  
  plt.plot(y_pos,num_data)
  # plt.xticks(y_pos,objects)
  # plt.ylabel('Label')
  plt.title('Distribution of train data (CD-t0)')
  plt.show()
  
  # * Meaning:
  # Data is biased to the the left region (small values)
