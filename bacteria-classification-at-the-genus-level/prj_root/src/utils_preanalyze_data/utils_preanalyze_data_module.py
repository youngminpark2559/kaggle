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
from scipy.stats import norm

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
from collections import Counter
from scipy.misc import imread
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from matplotlib import pyplot as plt
# %matplotlib inline

# ================================================================================
from src.utils import utils_image as utils_image
from src.utils import utils_common as utils_common

# ================================================================================
train_data="/mnt/1T-5e7/mycodehtml/bio_health/Diabetes/pima-indians-diabetes-database/Data/diabetes.csv"

# ================================================================================
def load_csv_file(path):
  loaded_csv=pd.read_csv(path,encoding='utf8')
  return loaded_csv

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

def estimated_PDF_by_using_mean_and_var(loaded_csv,args):
  # ================================================================================
  means=loaded_csv.describe().iloc[1,:]
  stds=loaded_csv.describe().iloc[2,:]
  vars=stds**2
  # print("means",means)
  # Pregnancies                 3.845052  
  # Glucose                     120.894531
  # BloodPressure               69.105469 
  # SkinThickness               20.536458 
  # Insulin                     79.799479 
  # BMI                         31.992578 
  # DiabetesPedigreeFunction    0.471876  
  # Age                         33.240885 
  # Outcome                     0.348958  

  # print("stds",stds)
  # Pregnancies                 3.369578  
  # Glucose                     31.972618 
  # BloodPressure               19.355807 
  # SkinThickness               15.952218 
  # Insulin                     115.244002
  # BMI                         7.884160  
  # DiabetesPedigreeFunction    0.331329  
  # Age                         11.760232 
  # Outcome                     0.476951  

  # print("vars",vars)
  # Pregnancies                 11.354056   
  # Glucose                     1022.248314 
  # BloodPressure               374.647271  
  # SkinThickness               254.473245  
  # Insulin                     13281.180078
  # BMI                         62.159984   
  # DiabetesPedigreeFunction    0.109779    
  # Age                         138.303046  
  # Outcome                     0.227483    

  params=list(zip(means,vars))
  # print("params",params)
  # [(3.8450520833333335, 11.354056320621417), (120.89453125, 1022.2483142519558), (69.10546875, 374.64727122718375), (20.536458333333332, 254.47324532811953), (79.79947916666667, 13281.180077955283), (31.992578124999977, 62.159983957382565), (0.4718763020833327, 0.10977863787313936), (33.240885416666664, 138.30304589037362), (0.3489583333333333, 0.22748261625380098)]

  # print("",len(params))
  # 9

  nb_h=len(params)/3
  nb_w=len(params)/3

  
  features=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
  for i,one_feat in enumerate(params):
    # print("one_feat",one_feat)
    # (3.8450520833333335, 3.3695780626988623)

    x_values=np.arange(-1000,1000,0.001)
    plt.subplot(nb_h,nb_w,i+1)
    plt.title(features[i]+" / mean: "+str(round(one_feat[0],2))+" / variance: "+str(round(one_feat[1],2)))
    plt.subplots_adjust(wspace=None,hspace=0.3)
    plt.plot(x_values,norm.pdf(x_values,one_feat[0],one_feat[1]))
  plt.show()
  # /home/young/Pictures/2019_05_10_06:48:15.png

def check_nan_vals(loaded_csv,args):
  print("loaded_csv.info()",loaded_csv.info())

  # Pregnancies                 768 non-null int64
  # Glucose                     768 non-null int64
  # BloodPressure               768 non-null int64
  # SkinThickness               768 non-null int64
  # Insulin                     768 non-null int64
  # BMI                         768 non-null float64
  # DiabetesPedigreeFunction    768 non-null float64
  # Age                         768 non-null int64
  # Outcome                     768 non-null int64

def see_correlations_on_features(loaded_csv,args):
  train_csv_wo_id=loaded_csv
  # print("train_csv_wo_id",train_csv_wo_id.shape)
  # (920, 5)

  # ================================================================================
  cor_mat=train_csv_wo_id.corr()
  # print("cor_mat",cor_mat.shape)
  # (9, 9)

  # print("cor_mat",cor_mat)
  #                           Pregnancies   Glucose  BloodPressure  SkinThickness  \
  # Pregnancies               1.000000     0.129459  0.141282      -0.081672        
  # Glucose                   0.129459     1.000000  0.152590       0.057328        
  # BloodPressure             0.141282     0.152590  1.000000       0.207371        
  # SkinThickness            -0.081672     0.057328  0.207371       1.000000        
  # Insulin                  -0.073535     0.331357  0.088933       0.436783        
  # BMI                       0.017683     0.221071  0.281805       0.392573        
  # DiabetesPedigreeFunction -0.033523     0.137337  0.041265       0.183928        
  # Age                       0.544341     0.263514  0.239528      -0.113970        
  # Outcome                   0.221898     0.466581  0.065068       0.074752        

  #                            Insulin       BMI  DiabetesPedigreeFunction  \
  # Pregnancies              -0.073535  0.017683 -0.033523                   
  # Glucose                   0.331357  0.221071  0.137337                   
  # BloodPressure             0.088933  0.281805  0.041265                   
  # SkinThickness             0.436783  0.392573  0.183928                   
  # Insulin                   1.000000  0.197859  0.185071                   
  # BMI                       0.197859  1.000000  0.140647                   
  # DiabetesPedigreeFunction  0.185071  0.140647  1.000000                   
  # Age                      -0.042163  0.036242  0.033561                   
  # Outcome                   0.130548  0.292695  0.173844                   

  #                                Age   Outcome  
  # Pregnancies               0.544341  0.221898  
  # Glucose                   0.263514  0.466581  
  # BloodPressure             0.239528  0.065068  
  # SkinThickness            -0.113970  0.074752  
  # Insulin                  -0.042163  0.130548  
  # BMI                       0.036242  0.292695  
  # DiabetesPedigreeFunction  0.033561  0.173844  
  # Age                       1.000000  0.238356  
  # Outcome                   0.238356  1.000000  

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

  # norm_corr_mat_df=pd.DataFrame(norm_corr_mat_np)
  norm_corr_mat_df=pd.DataFrame(cor_mat_np)
  # print("norm_corr_mat_df",norm_corr_mat_df)
  #           0         1         2
  # 0  1.000000  0.554199  0.209961
  # 1  0.554199  1.000000  0.000000
  # 2  0.209961  0.000000  1.000000

  new_col_name={0:"Pregnancies",1:"Glucose",2:"BloodPressure",3:"SkinThickness",4:"Insulin",5:"BMI",6:"DiabetesPedigreeFunction",7:"Age",8:"Outcome"}
  new_idx_name={0:"Pregnancies",1:"Glucose",2:"BloodPressure",3:"SkinThickness",4:"Insulin",5:"BMI",6:"DiabetesPedigreeFunction",7:"Age",8:"Outcome"}
  norm_corr_mat_df_a=norm_corr_mat_df.rename(columns=new_col_name,index=new_idx_name,inplace=False)
  # print("norm_corr_mat_df_a",norm_corr_mat_df_a)
  #                           Pregnancies   Glucose  BloodPressure  SkinThickness  \
  # Pregnancies               1.000000     0.218506  0.229004       0.028976        
  # Glucose                   0.218506     1.000000  0.239258       0.153687        
  # BloodPressure             0.229004     0.239258  1.000000       0.288330        
  # SkinThickness             0.028976     0.153687  0.288330       1.000000        
  # Insulin                   0.036255     0.399658  0.182129       0.494385        
  # BMI                       0.118103     0.300537  0.355225       0.454346        
  # DiabetesPedigreeFunction  0.072205     0.225464  0.139404       0.267334        
  # Age                       0.590820     0.338623  0.317383       0.000000        
  # Outcome                   0.301514     0.520996  0.160645       0.169312        

  #                           Insulin       BMI  DiabetesPedigreeFunction  \
  # Pregnancies               0.036255  0.118103  0.072205                   
  # Glucose                   0.399658  0.300537  0.225464                   
  # BloodPressure             0.182129  0.355225  0.139404                   
  # SkinThickness             0.494385  0.454346  0.267334                   
  # Insulin                   1.000000  0.279785  0.268311                   
  # BMI                       0.279785  1.000000  0.228516                   
  # DiabetesPedigreeFunction  0.268311  0.228516  1.000000                   
  # Age                       0.064392  0.134766  0.132324                   
  # Outcome                   0.219360  0.364990  0.258301                   

  #                                Age   Outcome  
  # Pregnancies               0.590820  0.301514  
  # Glucose                   0.338623  0.520996  
  # BloodPressure             0.317383  0.160645  
  # SkinThickness             0.000000  0.169312  
  # Insulin                   0.064392  0.219360  
  # BMI                       0.134766  0.364990  
  # DiabetesPedigreeFunction  0.132324  0.258301  
  # Age                       1.000000  0.316162  
  # Outcome                   0.316162  1.000000  

  sns.heatmap(norm_corr_mat_df_a)
  plt.show()
  # /home/young/Pictures/2019_05_09_21:27:16.png

  # Meaning:
  # 1.. Negative correlation: Insulin-Pregnancies, SkinThickness-Pregnancies, Age-SkinThickness,
  # 2.. Positive correlation: Age-Pregnancies, BMI-SkinThickness, Insluin-SkinThickness, Insulin-Glucose

def create_diabete_label_by_120_glucose(loaded_csv,args):
  label_col=np.zeros((loaded_csv.shape[0]))
  # print("label_col",label_col.shape)
  # (768,)

  glucose_data=loaded_csv.iloc[:,1]
  label_col[glucose_data>=120]=1
  # print("label_col",label_col)

  loaded_csv["glucose_over_120"]=label_col
  # print("loaded_csv",loaded_csv.head(2))
  #    Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI   DiabetesPedigreeFunction  Age  Outcome  glucose_over_120  
  # 0  6            148      72             35             0        33.6   0.627                     50   1        1.0               
  # 1  1            85       66             29             0        26.6   0.351                     31   0        0.0               

  return loaded_csv

def get_prior_probability(loaded_csv,args):
  # print("loaded_csv",loaded_csv.shape)
  # 768

  # print("loaded_csv.columns",loaded_csv.columns)
  # ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome', 'glucose_over_120']

  glucose_over_120_data=loaded_csv.iloc[:,-1]
  # print("glucose_over_120_data",glucose_over_120_data)
  # 0      1.0
  # 1      0.0

  num_high_glucose=glucose_over_120_data.sum()
  # print("num_high_glucose",num_high_glucose)
  # 360.0

  # ================================================================================
  # p(high_glucose|preg) = P(high_glucose \cap preg) / P(preg)

  
  

  # aa=pd.qcut(loaded_csv.iloc[:,0],[0.05, 0.1, 0.25, 0.5, 0.75, 1])
  # print("aa",aa)
  # afaf

  # new=pd.DataFrame()
  # for i in range(1,5):
  #   probB = loaded_csv.shift(i)['Pregnancies']
  #   print("probB",probB)

  #   probA = loaded_csv['glucose_over_120']
  #   new['prob -' + str(i)] = (probA * probB) / probB
  # # print("new",new)
  # afaf



  pregnancies_data=loaded_csv.iloc[:,0]
  # print("pregnancies_data",pregnancies_data)
  # 0      6 
  # 1      1 
  # 2      8 
  # 3      1 

  uniq_vals=np.unique(pregnancies_data)
  # print("uniq_vals",uniq_vals)
  # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 17]

  num_ele_in_one_group=int(len(uniq_vals)/4)
  # print("num_ele_in_one_group",num_ele_in_one_group)
  # 4

  # intervals
  # [0  1  2  3]
  # [4  5  6  7]
  # [8  9 10 11]
  # [12 13 14 15 17]

  mask_1st_interval=np.logical_and(0<=pregnancies_data,pregnancies_data<4)

  masked_loaded_csv=loaded_csv[mask_1st_interval]
  # print("masked_loaded_csv",masked_loaded_csv.shape)
  # (424, 10)

  masked_by_1_label=masked_loaded_csv[masked_loaded_csv.iloc[:,-1]==1.0]
  # print("masked_by_1_label",masked_by_1_label.shape)
  # (173, 10)

  # p(high_glucose|preg) = P(high_glucose \cap preg) / P(preg)

  P_high_glucose_AND_preg_0to3=173/768
  # print("P_high_glucose_AND_preg_0to3",P_high_glucose_AND_preg_0to3)
  # 0.22526041666666666

  P_preg_0to3=424/768
  # print("P_preg_0to3",P_preg_0to3)
  # 0.5520833333333334

  cond_prob_of_high_glucose_when_preg_0to3_is_given=P_high_glucose_AND_preg_0to3/P_preg_0to3
  # print("cond_prob_of_high_glucose_when_preg_0to3_is_given",cond_prob_of_high_glucose_when_preg_0to3_is_given)
  # 0.4














  loaded_csv_0_to_20=loaded_csv[int(len(loaded_csv)*0.0):int(len(loaded_csv)*0.2)]
  # print("loaded_csv_0_to_20",loaded_csv_0_to_20.head(2))
  #    Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \
  # 0  6            148      72             35             0        33.6   
  # 1  1            85       66             29             0        26.6   

  #    DiabetesPedigreeFunction  Age  Outcome  glucose_over_120  
  # 0  0.627                     50   1        1.0               
  # 1  0.351                     31   0        0.0               
  


  # print("new_pregnancies_data",new_pregnancies_data.shape)
  # (153,)


  








  nb_gt_120=np.sum(glucose_data>120)
  # print("nb_gt_120",nb_gt_120)
  # 349

  prob_val_of_glucose_gt_120=nb_gt_120/768
  # print("prob_val_of_glucose_gt_120",prob_val_of_glucose_gt_120)
  # 0.45


  BMI_data=loaded_csv.iloc[:,5]
  nb_gt_25_in_BMI=np.sum(BMI_data>25.0)
  # print("nb_gt_25_in_BMI",nb_gt_25_in_BMI)
  # 645

  prob_val_of_BMI_gt_25=nb_gt_25_in_BMI/768
  # print("prob_val_of_BMI_gt_25",prob_val_of_BMI_gt_25)
  # 0.83984375

  # Conditional probability
  # P[B|A]=P[A \cap B] \times P[A]
  # P[nb_gt_25_in_BMI|prob_val_of_BMI_gt_25]=P[prob_val_of_BMI_gt_25 \cap nb_gt_25_in_BMI] * P[prob_val_of_BMI_gt_25]
  # P[nb_gt_25_in_BMI|prob_val_of_BMI_gt_25]=P[prob_val_of_BMI_gt_25 \cap nb_gt_25_in_BMI] * 0.83
  # P[nb_gt_25_in_BMI|prob_val_of_BMI_gt_25]=(0.45*0.83984375)*0.83

  # ================================================================================
  # Bayes theorem
  # A: BMI_gt_25
  # B: glucose_gt_120

  # P[B|A] = (P[B|A]*P[A])/P[B]
  # P[B|A] = ((P[A \cap B]/P[B])*P[A]) / P[B]
  # (((0.45*0.84)/0.45)*0.84)/0.45

def get_actual_distribution_using_histogram(loaded_csv,args):
  columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
  preg_data=np.array(loaded_csv.iloc[:,0])
  # print("preg_data",preg_data)
  # [ 6  1  8  1  0  5  3 10  2  8  4 10 10  1  5  7  0  7  1  1  3  8  7  9

  for i in range(len(columns)):
    one_feat_data=np.array(loaded_csv.iloc[:,i])
    plt.subplot(len(columns)/3,len(columns)/3,i+1)
    plt.title(columns[i])
    plt.subplots_adjust(wspace=None,hspace=0.3)
    n,bins,patches=plt.hist(one_feat_data,bins=50)
  plt.show()

def see_pair_scatter_plot(loaded_csv,loaded_csv_label,args):
  # sns.pairplot(loaded_csv,diag_kind='hist')
  # plt.subplots_adjust(left=0.125,bottom=0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.2)
  # plt.show()

  # sns.pairplot(loaded_csv_label,diag_kind='hist',hue="glucose_over_120")
  # plt.subplots_adjust(left=0.125,bottom=0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.2)
  # plt.show()

  # sns.pairplot(loaded_csv_label,diag_kind='kde',hue="glucose_over_120",palette='bright') # pastel, bright, deep, muted, colorblind, dark
  # plt.title("With estimated distribution via Kernel Density Estimation")
  # plt.subplots_adjust(left=0.125,bottom=0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.2)
  # plt.show()

  # sns.pairplot(loaded_csv_label,diag_kind="kde",kind='reg',hue="glucose_over_120",palette='bright') # pastel, bright, deep, muted, colorblind, dark
  sns.pairplot(loaded_csv_label,kind='reg') # pastel, bright, deep, muted, colorblind, dark
  plt.title("Linear regression line which can explain distributional pattern of each 2 feature data")
  plt.subplots_adjust(left=0.125,bottom=0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.2)
  plt.show()
  # /home/young/Pictures/2019_05_10_08:05:39.png

  # Meaning
  # 1.. See Glucose-Insulin
  # - Less Glucose results in less Insulin
  # - Increase Glucose results in More Insulin
  # 2.. See BMI-Glucose
  # - Even if low BMI, it can have high Glucose
  # 3.. At least according this plotting, there is no strong correlation between all features and glucose level
  # - It means low BMI can have high glucose, high BMI can have high glucose.
  # - Low SkinThickness can have high glucose, high SkinThickness can have high glucose.

def analyze_data(args):
  afaf
  loaded_csv=load_csv_file(train_data)
  # print("loaded_csv",loaded_csv.shape)
  # (768, 9)

  # print("loaded_csv",loaded_csv.head(2))
  #    Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \
  # 0  6            148      72             35             0        33.6   
  # 1  1            85       66             29             0        26.6   

  #    DiabetesPedigreeFunction  Age  Outcome  
  # 0  0.627                     50   1        
  # 1  0.351                     31   0        

  # print("loaded_csv",loaded_csv.columns)
  # ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

  # print("loaded_csv",loaded_csv.head(2))
  #    Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \
  # 0  6            148      72             35             0        33.6   
  # 1  1            85       66             29             0        26.6   

  #    DiabetesPedigreeFunction  Age  Outcome  
  # 0  0.627                     50   1        
  # 1  0.351                     31   0        

  # ================================================================================
  # print("loaded_csv.describe()",loaded_csv.describe())
  #        Pregnancies     Glucose  BloodPressure  SkinThickness     Insulin  \
  # count  768.000000   768.000000  768.000000     768.000000     768.000000   
  # mean   3.845052     120.894531  69.105469      20.536458      79.799479    
  # std    3.369578     31.972618   19.355807      15.952218      115.244002   
  # min    0.000000     0.000000    0.000000       0.000000       0.000000     
  # 25%    1.000000     99.000000   62.000000      0.000000       0.000000     
  # 50%    3.000000     117.000000  72.000000      23.000000      30.500000    
  # 75%    6.000000     140.250000  80.000000      32.000000      127.250000   
  # max    17.000000    199.000000  122.000000     99.000000      846.000000   

  #               BMI  DiabetesPedigreeFunction         Age     Outcome  
  # count  768.000000  768.000000                768.000000  768.000000  
  # mean   31.992578   0.471876                  33.240885   0.348958    
  # std    7.884160    0.331329                  11.760232   0.476951    
  # min    0.000000    0.078000                  21.000000   0.000000    
  # 25%    27.300000   0.243750                  24.000000   0.000000    
  # 50%    32.000000   0.372500                  29.000000   0.000000    
  # 75%    36.600000   0.626250                  41.000000   1.000000    
  # max    67.100000   2.420000                  81.000000   1.000000    

  # ================================================================================
  # estimated_PDF_by_using_mean_and_var(loaded_csv,args)

  # ================================================================================
  # check_nan_vals(loaded_csv,args)
  
  # ================================================================================
  # see_correlations_on_features(loaded_csv,args)
 
  # ================================================================================
  loaded_csv_label=create_diabete_label_by_120_glucose(loaded_csv,args)
  
  # ================================================================================
  get_prior_probability(loaded_csv_label,args)
  afaf

  # ================================================================================
  # get_actual_distribution_using_histogram(loaded_csv,args)

  # ================================================================================
  see_pair_scatter_plot(loaded_csv,loaded_csv_label,args)
  