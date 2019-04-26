# conda activate py36gputorch041
# cd /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/utils/
# rm e.l && python utils_image.py 2>&1 | tee -a e.l && code e.l

import Augmentor
import numpy as np
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
import traceback
from skimage.transform import resize

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
    # print("paths_of_imgs",paths_of_imgs)
    # [('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/2fd37df8-bbb3-11e8-b2ba-ac1f6b6435d0_blue.png\n',
    #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/2fd37df8-bbb3-11e8-b2ba-ac1f6b6435d0_green.png\n',
    #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/2fd37df8-bbb3-11e8-b2ba-ac1f6b6435d0_red.png\n',
    #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/2fd37df8-bbb3-11e8-b2ba-ac1f6b6435d0_yellow.png\n'),
    #  ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/c5738370-bbaa-11e8-b2ba-ac1f6b6435d0_blue.png\n',
    #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/c5738370-bbaa-11e8-b2ba-ac1f6b6435d0_green.png\n',
    #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/c5738370-bbaa-11e8-b2ba-ac1f6b6435d0_red.png\n',
    #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/c5738370-bbaa-11e8-b2ba-ac1f6b6435d0_yellow.png\n')]

    labels_of_imgs=dataset_bs_paths[1]
    # print("labels_of_imgs",labels_of_imgs)
    # ['3 0' '25 0']

    labels_of_imgs_li=[]
    for one_lbl_set in labels_of_imgs:
      one_list=one_lbl_set.split(" ")
      # print("one_list",one_list)
      # ['13', '18']

      one_list=list(map(int,one_list))
      # print("one_list",one_list)
      # [13, 18]

      labels_of_imgs_li.append(one_list)

    # print("labels_of_imgs_li",labels_of_imgs_li)
    # [[13, 18], [22]]


    # ================================================================================
    # Load images
    dataset_bs_paths=[]
    for one_protein in paths_of_imgs:
      loaded_b_img=np.array(Image.open(one_protein[0].replace("\n","")))
      loaded_g_img=np.array(Image.open(one_protein[1].replace("\n","")))
      loaded_r_img=np.array(Image.open(one_protein[2].replace("\n","")))
      loaded_y_img=np.array(Image.open(one_protein[3].replace("\n","")))
      
      dataset_bs_paths.append([loaded_b_img,loaded_g_img,loaded_r_img,loaded_y_img])
      
    # print("dataset_bs_paths",np.array(dataset_bs_paths).shape)
    # (2, 4, 512, 512)

    # ================================================================================
    labels_of_imgs=labels_of_imgs_li

    # ================================================================================
    # region augmentation on group
    aug_pipeline=Augmentor.DataPipeline(dataset_bs_paths,labels_of_imgs)

    # ================================================================================
    # @ crop_by_size
    # aug_pipeline.crop_by_size(probability=1.0,width=48,height=48,centre=True)

    # ================================================================================
    # @ rotate
    aug_pipeline.rotate(probability=0.5,max_left_rotation=6,max_right_rotation=7)

    # ================================================================================
    # @ flip_random
    aug_pipeline.flip_random(probability=0.5)

    # ================================================================================
    # @ resize
    aug_pipeline.resize(probability=1.0,width=224,height=224,resample_filter="BILINEAR")

    # ================================================================================
    # @ sample
    sampled_trn_and_rgt_imgs_li,label_values=aug_pipeline.sample(int(args.batch_size))
    
    # print("sampled_trn_and_rgt_imgs_li",len(sampled_trn_and_rgt_imgs_li))
    # 2

    # print("label_values",label_values)
    # label_values [[18, 0], [22, 0, 21]]

    # ================================================================================
    sampled_trn_imgs=np.array(sampled_trn_and_rgt_imgs_li)/255.0
    # print("sampled_trn_imgs",sampled_trn_imgs.shape)
    # (2, 4, 224, 224)

    return sampled_trn_imgs,label_values

  except:
    print(traceback.format_exc())
    print("Error when loading images")
    print("path_to_be_loaded",path_to_be_loaded)
