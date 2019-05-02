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

# ================================================================================
def see_correlation_of_data(args):
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
  
  # ================================================================================
  # check_nan(train_csv)

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
  # plt.show()
  
  # ================================================================================
  # * Meaning
  # * Therea are negative and positive relationships between factors

  # ================================================================================
  
