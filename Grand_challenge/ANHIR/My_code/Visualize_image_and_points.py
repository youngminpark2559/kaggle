# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/bio_health/Grand_challenge/ANHIR/My_code && \
# rm e.l && python Visualize_image_and_points.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import pandas as pd
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
import matplotlib.pyplot as plt
import csv

# ================================================================================
def load_img(path):
  img=Image.open(path)
  arr=np.array(img)
  return arr

def load_csv_using_pandas(path):
  loaded_csv=pd.read_csv(path,encoding='utf8')
  return loaded_csv

def load_csv_to_list(path):
  with open(path,'r') as f:
    reader=csv.reader(f)
    your_list=list(reader)
  return your_list

# ================================================================================
breast_3_scale_20pc_HE="/mnt/1T-5e7/mycodehtml/bio_health/Grand_challenge/ANHIR/dataset_medium_images/breast_3/scale-20pc/HE.jpg"
breast_3_scale_20pc_HE_img=load_img(breast_3_scale_20pc_HE)
# print("breast_3_scale_20pc_HE_img",breast_3_scale_20pc_HE_img.shape)
# (11974, 17326, 3)

breast_3_scale_20pc_HE_list=load_csv_to_list("/mnt/1T-5e7/mycodehtml/bio_health/Grand_challenge/ANHIR/dataset_medium/breast_3/scale-20pc/HE.csv")

breast_3_scale_20pc_HE_np=np.array(breast_3_scale_20pc_HE_list)
# print("breast_3_scale_20pc_HE_np",breast_3_scale_20pc_HE_np)
# [['' 'X' 'Y']
#  ['0' '11237.2' '3608.8']

breast_3_scale_20pc_HE_np_processed=breast_3_scale_20pc_HE_np[1:,1:]
# print("breast_3_scale_20pc_HE_np_processed",breast_3_scale_20pc_HE_np_processed)
# [['11237.2' '3608.8']
#  ['11517.2' '3534.8']

breast_3_scale_20pc_HE_np_processed=np.asfarray(breast_3_scale_20pc_HE_np_processed,float)
# print("breast_3_scale_20pc_HE_np_processed",breast_3_scale_20pc_HE_np_processed)
# [[11237.2  3608.8]
#  [11517.2  3534.8]

x_vals=breast_3_scale_20pc_HE_np_processed[:,0]
# print("x_vals",x_vals)
# [11237.2 11517.2

y_vals=breast_3_scale_20pc_HE_np_processed[:,1]
# print("y_vals",y_vals)
# [ 3608.8  3534.8

plt.imshow(breast_3_scale_20pc_HE_img)
plt.scatter(x=x_vals,y=y_vals,c='g',s=10)
plt.show()
