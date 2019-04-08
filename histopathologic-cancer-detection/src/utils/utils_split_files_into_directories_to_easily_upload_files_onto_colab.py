# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/my_model/my_model_new_architecture/prj_root/src/utils && \
# rm e.l && python utils_split_files_into_directories_to_easily_upload_files_onto_colab.py \
# 2>&1 | tee -a e.l && code e.l

"""
This file can run separately to move big data files located in one directory 
into separated directories for the purpose of

1. you can see and access image files easily from separated directories
2. you can upload big data files onto Google drive easily, 
which will be used in Google colaboratory notebook
And after uploading zip files onto Google drive, 
I use 3rd party app called ZIP Extractor to decompress the zip files
3. or you can directly upload each directory (like /00, /01) onto Google drive 
without using zip files
"""

# ================================================================================
import sys,os
import math
from subprocess import call
import subprocess

# ================================================================================
def divisor_generator(n):
  large_divisors=[]
  for i in range(1,int(math.sqrt(n)+1)):
    if n%i==0:
      yield i
      if i*i!=n:
        large_divisors.append(n/i)
  for divisor in reversed(large_divisors):
      yield int(divisor)

def create_text_files_which_contains_all_paths_of_image_files():
  trn_dir={"trn_txt_file":"/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split"}
  test_dir={"test_txt_file":"/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/test"}

  # ================================================================================
  dir_list=[trn_dir,test_dir]
  for one_dir in dir_list:
    one_dir_name=list(one_dir.keys())[0]
    one_f_path=list(one_dir.values())[0]
    # print("one_dir_name",one_dir_name)
    # trn_txt_file
    # print("one_f_path",one_f_path)
    # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split

    path_where_txt_file_will_be_store="/".join(one_f_path.split("/")[:-1])
    # print("path_where_txt_file_will_be_store",path_where_txt_file_will_be_store)
    # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data

    # @ c cmd: command you will use
    cmd="find "+one_f_path+" | sort > "+path_where_txt_file_will_be_store+"/"+one_dir_name+".txt"
    # print("cmd",cmd)
    # find /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train | sort > /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/trn_txt_file.txt

    # @ Generate text file which contains all paths of images
    call(cmd,shell=True)

def split_train_imgs_into_multiple_dirs():
  trn_txt_file_path="/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/trn_txt_file.txt"

  # @ Load text file into list
  lines_list=[]
  with open(trn_txt_file_path) as f:
    lines=f.readlines()
    lines_list.extend(lines)
  # print("lines_list",lines_list)
  # ['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train\n',
  
  # @ Remove useless paths
  result=[]
  for one_path in lines_list:
    if (".tif" not in one_path):
      one_path=None
      result.append(one_path)
    else:
      result.append(one_path)
  processed_paths=[x for x in result if x is not None]
  # print("processed_paths",processed_paths)

  num_img=len(processed_paths)
  # print("num_img",num_img)
  # 220025
  
  assert num_img==220025,'Some files are missing in your text file which is supposed to contain all paths of train images'
  
  # @ Let's find proper candidate divisors which can perfectly divide 220025
  # print(list(divisor_generator(220025)))
  # [1, 5, 13, 25, 65, 325, 677, 3385, 8801, 16925, 44005, 220025]
  # 220025/65=3385.0
  # I will create 65 groups that each group should have 3385 images
  # print("len(processed_paths)",len(processed_paths))
  # 220025

  processed_paths_iter=iter(processed_paths)
  entire_grp=[]
  # @ Create 65 groups into one list
  for i in range(65):
    one_grp=[]
    # Get 3385 number of paths
    for j in range(3385):
      one_path=next(processed_paths_iter)
      one_grp.append(one_path)
    # print("len(one_grp)",len(one_grp))
    # 3385
    entire_grp.append(one_grp)
  # print("len(entire_grp)",len(entire_grp))
  # 65
  i=0

  # @ Iterate all groups
  for one_grp in entire_grp:
    # print("one_grp",one_grp)
    # ['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/00001b2b5609af42ab0ab276dd4cd41c3e7745b5.tif\n',
    # ...]

    # @ Iterate one groups
    for one_path in one_grp:
      base_file_name=one_path.split("/")[-1]
      # print("base_file_name",base_file_name)
      # 00001b2b5609af42ab0ab276dd4cd41c3e7745b5.tif
      # print("one_path",one_path)
      # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/00001b2b5609af42ab0ab276dd4cd41c3e7745b5.tif

      # @ Convert 0 to 00, 1 to 01
      if len(str(i))==1:
        dir_num=str(0)+str(i)
      dir_name="/".join(one_path.split("/")[:-2])+"/train_split"+"/"+str(i)
      # print("dir_name",dir_name)
      # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/0

      # @ Create directory
      if not os.path.exists(dir_name):
        os.makedirs(dir_name)
      cmd=("cp "+one_path+" "+dir_name+"/"+base_file_name).replace("\n","")
      # print("cmd",cmd)
      # cp /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/00001b2b5609af42ab0ab276dd4cd41c3e7745b5.tif /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/0/00001b2b5609af42ab0ab276dd4cd41c3e7745b5.tif
      call(cmd,shell=True)

    i=i+1 

def create_text_files_from_split_directories():
  trn_split_dir={"trn_txt_file":"/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split"}
  dir_list=[trn_split_dir]

  for one_dir in dir_list:
    # @ key name like trn_txt_file
    # @ and its value like /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split
    one_dir_name=list(one_dir.keys())[0]
    one_f_path=list(one_dir.values())[0]
    # print("one_dir_name",one_dir_name)
    # trn_txt_file
    # print("one_f_path",one_f_path)
    # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split

    # @ Path where you will store text file which stores all paths of images
    path_where_txt_file_will_be_store=one_f_path
    # print("path_where_txt_file_will_be_store",path_where_txt_file_will_be_store)
    # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split

    # @ c cmd: prepare command you will use
    cmd="find "+one_f_path+" | sort > "+path_where_txt_file_will_be_store+"/"+one_dir_name+".txt"
    # print("cmd",cmd)
    # find /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split | sort > /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/trn_txt_file.txt

    # @ Generate text file which contains all paths of images
    call(cmd,shell=True)

def remove_useless_paths_from_text_file_containing_split_paths(txt_file_path):
  trn_txt_file_path=txt_file_path

  # Load text file into list
  lines_list=[]
  with open(trn_txt_file_path) as f:
    lines=f.readlines()
    lines_list.extend(lines)
  # print("lines_list",lines_list)
  # ['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train\n',

  # @ Remove useless paths
  result=[]
  for one_path in lines_list:
    if (".tif" not in one_path):
      one_path=None
      result.append(one_path)
    else:
      result.append(one_path)
  processed_paths=[x for x in result if x is not None]
  # print("processed_paths",processed_paths)

  num_img=len(processed_paths)
  print("num_img",num_img)
  # 220025

  assert num_img==220025,'Some files are missing in your text file which is supposed to contain all paths of train images'

  with open(trn_txt_file_path.replace("trn_txt_file","trn_txt_file_processed"),'w') as f:
    for item in processed_paths:
      f.write(item)

def change_local_img_path_into_colab_img_path(txt_file_path,dir_for_cloud):
  # dir_for_cloud="/content/drive/My Drive/Kaggle_pathology/Data/train"

  # ================================================================================
  # @ Load contents of text file
  lines_list=[]
  with open(txt_file_path) as f:
    lines=f.readlines()
    lines_list.extend(lines)
  # print("lines_list",lines_list)
  # ['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/00/00001b2b5609af42ab0ab276dd4cd41c3e7745b5.tif\n',

  # ================================================================================
  # @ Change path

  to_be_txt=[]
  for one_path in lines_list:
    one_path=one_path.replace(
      "/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split",
      dir_for_cloud)
    to_be_txt.append(one_path)
  # print("to_be_txt",to_be_txt)

  # ================================================================================
  # @ Write change changed contents into text file

  with open("/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/trn_txt_file_processed_colab.txt",'w') as w_stream:
    for one_path in to_be_txt:
      # print("one_path",one_path)
      # /content/drive/My Drive/ML/IID/dataset/text_based/temp_for_colab/cgmit_ori/0001/CGINTRINSICS_0001_000000_mlt.png
      w_stream.write(one_path)

def compress_each_directory_by_zip():
  base_dir="/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split"
  # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/00

  # @ Iterate 65 directories
  for i in range(65):
    if len(str(i))==1:
      dir_num=str(0)+str(i)
    else:
      dir_num=str(i)
    cmd="cd "+base_dir+"&&zip -r "+dir_num+".zip ./"+dir_num
    # print("cmd",cmd)
    # cd /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split&&zip -r 00.zip ./00
    # @ Execute zip
    call(cmd,shell=True)

# @ Execute these in turn
# create_text_files_which_contains_all_paths_of_image_files()
# split_train_imgs_into_multiple_dirs()
# create_text_files_from_split_directories()
remove_useless_paths_from_text_file_containing_split_paths(txt_file_path="/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/trn_txt_file.txt")
change_local_img_path_into_colab_img_path(
  txt_file_path="/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/trn_txt_file_processed.txt",
  dir_for_cloud="/content/drive/My Drive/Kaggle_pathology/Data/train")
# compress_each_directory_by_zip()
