# conda activate py36gputorch041
# cd /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/utils/
# rm e.l && python utils_image.py 2>&1 | tee -a e.l && code e.l

import Augmentor
import numpy as np
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
import traceback

# ================================================================================
def use_augmetor_for_tumor_data(dataset_bs_paths,args):
  """
  Act
    * 
  
  Params
    * dataset_bs_paths
    * args
  
  Return
    * 
  """
  try:
    path_to_be_loaded=dataset_bs_paths

    paths_of_imgs=list(dataset_bs_paths[0])
    labels_of_imgs=dataset_bs_paths[1]
    # print("paths_of_imgs",paths_of_ismgs)
    # print("labels_of_imgs",labels_of_imgs)
    # ['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/fc6fd52b6f1e57be88a6d15018d2bcda2c2337e0.tif\n',
    #  '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/319ed57986f823a10a02674776c37ad1f1045b5d.tif\n',
    #  '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/c4ad6de9bbd4a91d8ee84728db60ed4087796211.tif\n',
    #  '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/81d1c7a84b5a537de928de3d9465d57bb373309d.tif\n',
    #  '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/d5e22bf6be37cd4619a26a69bcc28b3ec801f5b5.tif\n']
    # [1 0 0 0 0]

    # ================================================================================
    # Load images
    dataset_bs_paths=[]
    for x in paths_of_imgs:
      loaded_img=Image.open(x.replace("\n",""))
      # WHen you don't use gt image
      loaded_img=np.array(loaded_img)
      dataset_bs_paths.append([loaded_img])
    # print("dataset_bs_paths",len(dataset_bs_paths))
    # print("dataset_bs_paths",dataset_bs_paths)
    # 5
    # [array([[[ 56,  44, 108],
    #          [ 60,  38, 121],
    #          [ 59,  28, 132],

    # ================================================================================
    # I'm not going to use aggressive data augmentation on tumor dataset
    # just will use rotation, flip_random

    # print("dataset_bs_paths",np.array(dataset_bs_paths).shape)
    # dataset_bs_paths (5, 2, 96, 96, 3)

    labels_of_imgs=labels_of_imgs.tolist()
    # print("labels_of_imgs",type(labels_of_imgs))
    # print("labels_of_imgs",labels_of_imgs)
    # labels_of_imgs [0, 1, 0, 0, 0]

    aug_pipeline=Augmentor.DataPipeline(dataset_bs_paths,labels_of_imgs)

    # ================================================================================
    # crop_by_size
    aug_pipeline.crop_by_size(probability=1.0,width=48,height=48,centre=True)

    # ================================================================================
    # rotate
    aug_pipeline.rotate(probability=0.5,max_left_rotation=6,max_right_rotation=7)

    # ================================================================================
    # flip_random
    aug_pipeline.flip_random(probability=0.5)

    # ================================================================================
    # Random sample images
    sampled_trn_and_rgt_imgs_li,label_values=aug_pipeline.sample(int(args.batch_size))
    # print("sampled_trn_and_rgt_imgs_li",len(sampled_trn_and_rgt_imgs_li))
    # sampled_trn_and_rgt_imgs_li 5

    # print("label_values",label_values)
    # label_values [0, 0, 0, 0, 1]

    # --------------------------------------------------------------------------------
    sampled_trn_and_rgt_imgs=np.array(sampled_trn_and_rgt_imgs_li)/255.0
    # print("sampled_trn_and_rgt_imgs",sampled_trn_and_rgt_imgs.shape)
    # sampled_trn_and_rgt_imgs (5, 1, 64, 64, 3)

    sampled_trn_imgs=sampled_trn_and_rgt_imgs[:,0,:,:,:]

    sampled_trn_imgs_tc=sampled_trn_imgs.transpose(0,3,1,2)
    # print("sampled_trn_imgs_tc",sampled_trn_imgs_tc.shape)
    # (11, 3, 96, 96)

    return sampled_trn_imgs_tc,label_values

  except:
    print(traceback.format_exc())
    print("Error when loading images")
    print("path_to_be_loaded",path_to_be_loaded)
