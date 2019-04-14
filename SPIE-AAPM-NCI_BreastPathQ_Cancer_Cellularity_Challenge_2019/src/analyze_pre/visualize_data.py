# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/prj_root/src/utils && \
# rm e.l && python utils_image.py 2>&1 | tee -a e.l && code e.l

import matplotlib as mpl
from PIL import Image
import PIL.ImageOps
import cv2
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import image
import timeit
import sys
np.set_printoptions(threshold=sys.maxsize)
import scipy.misc
from skimage.transform import resize
from scipy.ndimage import convolve
from skimage.restoration import denoise_tv_chambolle
from skimage.morphology import square
import skimage
import scipy.ndimage.morphology

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
from torchvision import transforms
from torch.autograd import Variable

# ================================================================================
from src.utils import utils_common as utils_common
from src.utils import utils_image as utils_image
from src.utils import utils_data as utils_data

# ================================================================================
def visualize(args):
  img_crop_path="/mnt/1T-5e7/mycodehtml/bio_health/SPIE-AAPM-NCI_BreastPathQ_Cancer_Cellularity_Challenge_2019/Data/breastpathq/datasets/cells/1_Region 1_crop.tif"
  loaded_img_crop=utils_image.load_img(img_crop_path)
  # print("loaded_img",loaded_img.shape)
  # (451, 521, 3)

  xml_key_path="/mnt/1T-5e7/mycodehtml/bio_health/SPIE-AAPM-NCI_BreastPathQ_Cancer_Cellularity_Challenge_2019/Data/breastpathq/datasets/cells/1_Region 1_key.xml"
  whole_xml_key_data=utils_data.load_xml_file(xml_key_path)

  img_mask_path="/mnt/1T-5e7/mycodehtml/bio_health/SPIE-AAPM-NCI_BreastPathQ_Cancer_Cellularity_Challenge_2019/Data/breastpathq/datasets/cells/1_Region 1_mask.tif"
  loaded_img_mask=utils_image.load_img(img_mask_path)

  xml_crop_path="/mnt/1T-5e7/mycodehtml/bio_health/SPIE-AAPM-NCI_BreastPathQ_Cancer_Cellularity_Challenge_2019/Data/breastpathq/datasets/cells/Sedeen/1_Region 1_crop.session.xml"
  whole_xml_crop_data=utils_data.load_xml_file(xml_crop_path)

  # ================================================================================
  # print("whole_xml_key_data",whole_xml_key_data)

  data_name=whole_xml_key_data[0][0]
  data_region=whole_xml_key_data[0][1]
  data_color=whole_xml_key_data[0][2]
  data_points=whole_xml_key_data[0][3]
  # print("data_points",dir(data_points))

  data_points_li=str(data_points).replace("\n","").replace("<point-list>","").replace("<point>","").replace("</point-list>","").split("</point>")[:-1]
  # print("data_points_li",data_points_li)
  # ['32,18', '51,13',

  x_vals=[]
  y_vals=[]
  for one_coord in data_points_li:
    x_val=one_coord.split(",")[0]
    y_val=one_coord.split(",")[1]
    x_vals.append(x_val)
    y_vals.append(y_val)

  # print("x_vals",x_vals)
  # ['32', '51',
  # print("y_vals",y_vals)
  # ['18', '13',

  x_vals=list(map(int,x_vals))
  y_vals=list(map(int,y_vals))
  # print("x_vals",x_vals)
  # [32, 51,
  # print("y_vals",y_vals)
  # [18, 13,

  # ================================================================================
  data_name_1=whole_xml_key_data[1][0]
  data_region_1=whole_xml_key_data[1][1]
  data_color_1=whole_xml_key_data[1][2]
  data_points_1=whole_xml_key_data[1][3]
  # print("data_points_1",data_points_1)

  data_points_1_li=str(data_points_1).replace("\n","").replace("<point-list>","").replace("<point>","").replace("</point-list>","").split("</point>")[:-1]
  # print("data_points_1_li",data_points_1_li)
  # ['32,18', '51,13',

  x_vals_1=[]
  y_vals_1=[]
  for one_coord in data_points_1_li:
    x_val_1=one_coord.split(",")[0]
    y_val_1=one_coord.split(",")[1]
    x_vals_1.append(x_val_1)
    y_vals_1.append(y_val_1)

  # print("x_vals_1",x_vals_1)
  # ['32', '51',
  # print("y_vals_1",y_vals_1)
  # ['18', '13',

  x_vals_1=list(map(int,x_vals_1))
  y_vals_1=list(map(int,y_vals_1))
  # print("x_vals_1",x_vals_1)
  # [32, 51,
  # print("y_vals_1",y_vals_1)
  # [18, 13,
  # fafaf

  # ================================================================================
  data_name_2=whole_xml_key_data[2][0]
  data_region_2=whole_xml_key_data[2][1]
  data_color_2=whole_xml_key_data[2][2]
  data_points_2=whole_xml_key_data[2][3]
  # print("data_points_2",dir(data_points_2))

  data_points_2_li=str(data_points_2).replace("\n","").replace("<point-list>","").replace("<point>","").replace("</point-list>","").split("</point>")[:-1]
  # print("data_points_2_li",data_points_2_li)
  # ['32,18', '51,13',

  x_vals_2=[]
  y_vals_2=[]
  for one_coord in data_points_2_li:
    x_val_2=one_coord.split(",")[0]
    y_val_2=one_coord.split(",")[1]
    x_vals_2.append(x_val_2)
    y_vals_2.append(y_val_2)

  # print("x_vals_2",x_vals_2)
  # ['32', '51',
  # print("y_vals_2",y_vals_2)
  # ['18', '13',

  x_vals_2=list(map(int,x_vals_2))
  y_vals_2=list(map(int,y_vals_2))
  # print("x_vals_2",x_vals_2)
  # [32, 51,
  # print("y_vals_2",y_vals_2)
  # [18, 13,

  # ================================================================================
  utils_image.scatter_points_onto_img(loaded_img_crop,x_vals,y_vals,color="b",title="Image: 1_Region 1_crop.tif, XML: 1_Region 1_key.xml Region 2")
  # /home/young/Pictures/2019_04_14_01:14:03.png
  utils_image.scatter_points_onto_img(loaded_img_crop,x_vals_1,y_vals_1,color="g",title="Image: 1_Region 1_crop.tif, XML: 1_Region 1_key.xml Region 3")
  # /home/young/Pictures/2019_04_14_01:14:35.png
  utils_image.scatter_points_onto_img(loaded_img_crop,x_vals_2,y_vals_2,color="r",title="Image: 1_Region 1_crop.tif, XML: 1_Region 1_key.xml Region 4")
  # /home/young/Pictures/2019_04_14_01:14:58.png

  # ================================================================================
  # print(np.unique(loaded_img_mask))
  # [  0  85 127 255]
  # print("loaded_img_mask",loaded_img_mask[:,:,:3])
  unique_colors=list(set( tuple(v) for m2d in loaded_img_mask[:,:,:3] for v in m2d ))
  # print("unique_colors",unique_colors)
  # [(0, 255, 127), (0, 0, 0), (255, 255, 255), (0, 85, 0)]

  def use_mask_tif():
    for one_color in unique_colors:
      # print("one_color",one_color)
      # (0, 255, 127)
      # new_img=np.where(loaded_img_mask[:,:,:3]==np.array(one_color),1.0,0.0)

      loaded_img_mask_proc=loaded_img_mask[:,:,:3]

      # (0, 255, 127) locations are True
      new_img=np.all(loaded_img_mask_proc==np.array(one_color), axis=2)
      # print("new_img",new_img.shape)
      # print("new_img",new_img)

      new_img=skimage.morphology.binary_dilation(new_img,square(3))
      new_img=new_img[:,:,np.newaxis].astype("uint8")
      # None (0, 255, 127) locations are 1
      new_img=np.where(new_img==0,1,0)
      # print("new_img",new_img)

      masked_img=loaded_img_crop*new_img
      stacked=np.vstack(masked_img)
      # # print("stacked",stacked.shape)
      # # (234971, 3)
      # # afaf
      
      # # stacked=loaded_img_crop_mask.reshape((-1,3))
      # # print("stacked",stacked.shape)
      # # (234971, 3)

      idx=np.all(stacked==[0,0,0],1)
      # # print("idx",idxs.shape)
      # # idx (234971,)
      # # print("idx",idx)
      # # afaf

      # a1=np.vstack(loaded_img_crop)

      stacked[idx]=[0,255,0]
      # # print("a1",a1.shape)
      # # a1 (234971, 3)

      loaded_img_crop_new=stacked.reshape(loaded_img_crop.shape[0],loaded_img_crop.shape[1],3)
      # print("loaded_img_crop",loaded_img_crop.shape)


      # loaded_img_crop_masked=loaded_img_crop[loaded_img_crop_mask]
      # print("loaded_img_crop_masked",loaded_img_crop_masked)
      # # loaded_img_crop[loaded_img_crop_mask] (18, 3)
      # # loaded_img_crop[loaded_img_crop_mask] (387, 3)
      # # loaded_img_crop[loaded_img_crop_mask] (234971, 3)
      # # loaded_img_crop[loaded_img_crop_mask] (1629, 3)
      # loaded_img_crop[loaded_img_crop_mask]=[0,250,0]

      # # print("loaded_img_crop",loaded_img_crop)
      
      # plt.imshow(loaded_img_crop*nesw_img)
      plt.imshow(loaded_img_crop_new)
      plt.title("File: 1_Region 1_mask.tif, "+str(one_color))
      # /home/young/Pictures/2019_04_14_10:01:03.png
      # /home/young/Pictures/2019_04_14_10:01:18.png
      # /home/young/Pictures/2019_04_14_10:01:34.png

      plt.show()
  
  use_mask_tif()

  # ================================================================================
  # print("whole_xml_crop_data",whole_xml_crop_data)
  
  data_name=whole_xml_crop_data[0][0]
  data_region=whole_xml_crop_data[0][1]
  data_color=whole_xml_crop_data[0][2]
  data_points=whole_xml_crop_data[0][3]
  # print("data_points",dir(data_points))

  data_points_li=str(data_points).replace("\n","").replace("<point-list>","").replace("<point>","").replace("</point-list>","").split("</point>")[:-1]
  # print("data_points_li",data_points_li)
  # ['32,18', '51,13',

  x_vals=[]
  y_vals=[]
  for one_coord in data_points_li:
    x_val=one_coord.split(",")[0]
    y_val=one_coord.split(",")[1]
    x_vals.append(x_val)
    y_vals.append(y_val)

  # print("x_vals",x_vals)
  # ['32', '51',
  # print("y_vals",y_vals)
  # ['18', '13',

  x_vals=list(map(int,x_vals))
  y_vals=list(map(int,y_vals))
  # print("x_vals",x_vals)
  # [32, 51,
  # print("y_vals",y_vals)
  # [18, 13,

  # ================================================================================
  data_name_1=whole_xml_crop_data[1][0]
  data_region_1=whole_xml_crop_data[1][1]
  data_color_1=whole_xml_crop_data[1][2]
  data_points_1=whole_xml_crop_data[1][3]
  # print("data_points_1",data_points_1)
  # afaf

  data_points_1_li=str(data_points_1).replace("\n","").replace("<point-list>","").replace("<point>","").replace("</point-list>","").split("</point>")[:-1]
  # print("data_points_1_li",data_points_1_li)
  # ['32,18', '51,13',

  x_vals_1=[]
  y_vals_1=[]
  for one_coord in data_points_1_li:
    x_val_1=one_coord.split(",")[0]
    y_val_1=one_coord.split(",")[1]
    x_vals_1.append(x_val_1)
    y_vals_1.append(y_val_1)

  # print("x_vals_1",x_vals_1)
  # ['32', '51',
  # print("y_vals_1",y_vals_1)
  # ['18', '13',

  x_vals_1=list(map(int,x_vals_1))
  y_vals_1=list(map(int,y_vals_1))
  # print("x_vals_1",x_vals_1)
  # [32, 51,
  # print("y_vals_1",y_vals_1)
  # [18, 13,
  # fafaf

  # ================================================================================
  data_name_2=whole_xml_crop_data[2][0]
  data_region_2=whole_xml_crop_data[2][1]
  data_color_2=whole_xml_crop_data[2][2]
  data_points_2=whole_xml_crop_data[2][3]
  # print("data_points_2",dir(data_points_2))

  data_points_2_li=str(data_points_2).replace("\n","").replace("<point-list>","").replace("<point>","").replace("</point-list>","").split("</point>")[:-1]
  # print("data_points_2_li",data_points_2_li)
  # ['32,18', '51,13',

  x_vals_2=[]
  y_vals_2=[]
  for one_coord in data_points_2_li:
    x_val_2=one_coord.split(",")[0]
    y_val_2=one_coord.split(",")[1]
    x_vals_2.append(x_val_2)
    y_vals_2.append(y_val_2)

  # print("x_vals_2",x_vals_2)
  # ['32', '51',
  # print("y_vals_2",y_vals_2)
  # ['18', '13',

  x_vals_2=list(map(int,x_vals_2))
  y_vals_2=list(map(int,y_vals_2))
  # print("x_vals_2",x_vals_2)
  # [32, 51,
  # print("y_vals_2",y_vals_2)
  # [18, 13,

  # ================================================================================
  utils_image.scatter_points_onto_img(loaded_img_crop,x_vals,y_vals,color="b",title="Image: 1_Region 1_crop.tif, XML: 1_Region 1_crop.session.xml Region 2")
  # /home/young/Pictures/2019_04_14_10:02:20.png
  utils_image.scatter_points_onto_img(loaded_img_crop,x_vals_1,y_vals_1,color="g",title="Image: 1_Region 1_crop.tif, XML: 1_Region 1_crop.session.xml Region 3")
  # /home/young/Pictures/2019_04_14_10:02:44.png
  utils_image.scatter_points_onto_img(loaded_img_crop,x_vals_2,y_vals_2,color="r",title="Image: 1_Region 1_crop.tif, XML: 1_Region 1_crop.session.xml Region 4")
  # /home/young/Pictures/2019_04_14_10:02:57.png


