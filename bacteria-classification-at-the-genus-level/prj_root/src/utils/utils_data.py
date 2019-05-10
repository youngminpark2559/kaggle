# conda activate py36gputorch041
# cd /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/utils/
# rm e.l && python utils_image.py 2>&1 | tee -a e.l && code e.l

import Augmentor
import numpy as np
import pandas as pd
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
import traceback
import warnings
from skimage.transform import resize
from sklearn.utils import resample
from sklearn.utils import shuffle

# ================================================================================
from src.utils import utils_common as utils_common
from src.utils import utils_image as utils_image

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
    # ['/mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level/Train_Images/1529.png\n', '/mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level/Train_Images/0580.png\n']

    labels_of_imgs=dataset_bs_paths[1]
    # print("labels_of_imgs",labels_of_imgs)
    # ['salmonella' 'salmonella']

    # ================================================================================
    # Load images
    dataset_bs_paths=[]
    for x in paths_of_imgs:
      loaded_img=Image.open(x.replace("\n",""))

      # When you don't use gt image
      loaded_img=np.array(loaded_img)
      # print("loaded_img",loaded_img.shape)
      # (1024, 1280)

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
    # ['staphylococus', 'salmonella']

    # region augmentation on group
    aug_pipeline=Augmentor.DataPipeline(dataset_bs_paths,labels_of_imgs)

    # ================================================================================
    # crop_by_size
    # aug_pipeline.crop_by_size(probability=1.0,width=48,height=48,centre=True)

    aug_pipeline.resize(probability=1.0,width=224,height=224,resample_filter="BILINEAR")

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
    # 2

    # print("label_values",label_values)
    # ['salmonella', 'salmonella']

    # --------------------------------------------------------------------------------
    sampled_trn_and_rgt_imgs=np.array(sampled_trn_and_rgt_imgs_li)/255.0
    # print("sampled_trn_and_rgt_imgs",sampled_trn_and_rgt_imgs.shape)
    # (2, 1, 224, 224)

    sampled_trn_imgs=sampled_trn_and_rgt_imgs

    sampled_trn_imgs_tc=sampled_trn_imgs
    # print("sampled_trn_imgs_tc",sampled_trn_imgs_tc.shape)
    # (2, 1, 224, 224)
    # endregion 
    
    # # ================================================================================
    # # c kind_of_DA: you create list which contains kind of data augmentation
    # # kind_of_DA=["no_DA","ud","lr","p3","p6","p9","n3","n6","n9"]
    # kind_of_DA=["no_DA","ud","lr","p3","p6","n3","n6"]

    # # c chosen_DA: you get chosen kind of data augmentation
    # chosen_DA=np.random.choice(kind_of_DA,1,replace=False)[0]
    # # print("chosen_DA",chosen_DA)
    # # lr

    # dataset_bs_img_np=np.array(dataset_bs_paths).squeeze()
    # # print("dataset_bs_img_np",dataset_bs_img_np.shape)
    # # (10, 96, 96, 3)

    # # ================================================================================
    # # @ Flip or rotate

    # after_aug_imgs=[]
    # for one_idx in range(dataset_bs_img_np.shape[0]):
    #   one_img=dataset_bs_img_np[one_idx,:,:,:]
    #   # print("one_img",one_img.shape)
    #   # (48, 48, 3)
    #   # scipy.misc.imsave('./tumor_before_DA_'+str(one_idx)+'.png',one_img)

    #   if chosen_DA=="ud":
    #     one_img=np.flipud(one_img)
    #   elif chosen_DA=="lr":
    #     one_img=np.fliplr(one_img)
    #   elif chosen_DA=="p3":
    #     one_img=scipy.ndimage.interpolation.rotate(one_img,angle=3,reshape=True,mode="reflect")
    #   elif chosen_DA=="p6":
    #     one_img=scipy.ndimage.interpolation.rotate(one_img,angle=6,reshape=True,mode="reflect")
    #   elif chosen_DA=="p9":
    #     one_img=scipy.ndimage.interpolation.rotate(one_img,angle=9,reshape=True,mode="reflect")
    #   elif chosen_DA=="n3":
    #     one_img=scipy.ndimage.interpolation.rotate(one_img,angle=-3,reshape=True,mode="reflect")
    #   elif chosen_DA=="n6":
    #     one_img=scipy.ndimage.interpolation.rotate(one_img,angle=-6,reshape=True,mode="reflect")
    #   elif chosen_DA=="n9":
    #     one_img=scipy.ndimage.interpolation.rotate(one_img,angle=-9,reshape=True,mode="reflect")
    #   else:
    #       pass
      
    #   # ================================================================================
    #   one_img=np.clip(one_img/255.0,0.,1.)

    #   # ================================================================================
    #   # @ Resize image to (224,224,3)

    #   one_img=resize(one_img,(224,224))
    #   # print("one_img",one_img.shape)
    #   # (224, 224, 3)

    # ================================================================================
    # scipy.misc.imsave('./tumor_after_DA_'+str(chosen_DA)+str(one_idx)+'.png',one_img)

    # after_aug_imgs.append(one_img)

    # after_aug_imgs_np=np.array(after_aug_imgs)

    # # ================================================================================
    # # @ Center crop

    # # after_aug_imgs_np=after_aug_imgs_np[:,24:72,24:72,:]
    # # print("after_aug_imgs_np",after_aug_imgs_np.shape)
    # # (10, 48, 48, 3)

    # # ================================================================================
    # after_aug_imgs_np=after_aug_imgs_np.transpose(0,3,1,2)
    # # print("after_aug_imgs_np",after_aug_imgs_np.shape)
    # # (40, 3, 224, 224)

    # # ================================================================================
    # # print("labels_of_imgs",labels_of_imgs)
    # # [1 1 1 0 0 0 0 0 0 0]

    # labels_of_imgs_np=np.array(labels_of_imgs).astype("float32")
    # # print("labels_of_imgs_np",labels_of_imgs_np)

    return sampled_trn_imgs_tc,label_values

  except:
    print(traceback.format_exc())
    print("Error when loading images")
    print("path_to_be_loaded",path_to_be_loaded)


# ================================================================================
def create_batch_pair_of_paths(data,args):
  # print("data",data)
  # [[('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/5061784c-bbca-11e8-b2bc-ac1f6b6435d0_blue.png',
  #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/89eab306-bbbd-11e8-b2ba-ac1f6b6435d0_blue.png'),
  #   ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/5061784c-bbca-11e8-b2bc-ac1f6b6435d0_green.png',
  #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/89eab306-bbbd-11e8-b2ba-ac1f6b6435d0_green.png'),
  #   ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/5061784c-bbca-11e8-b2bc-ac1f6b6435d0_red.png',
  #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/89eab306-bbbd-11e8-b2ba-ac1f6b6435d0_red.png'),
  #   ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/5061784c-bbca-11e8-b2bc-ac1f6b6435d0_yellow.png',
  #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/89eab306-bbbd-11e8-b2ba-ac1f6b6435d0_yellow.png')],
  #  ('3','19 0')]
  
  # c paths_of_imgs: paths of images
  paths_of_imgs=data[0]
  # print("paths_of_imgs",paths_of_imgs)
  # [('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/5061784c-bbca-11e8-b2bc-ac1f6b6435d0_blue.png',
  #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/89eab306-bbbd-11e8-b2ba-ac1f6b6435d0_blue.png'),
  #  ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/5061784c-bbca-11e8-b2bc-ac1f6b6435d0_green.png',
  #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/89eab306-bbbd-11e8-b2ba-ac1f6b6435d0_green.png'),
  #  ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/5061784c-bbca-11e8-b2bc-ac1f6b6435d0_red.png',
  #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/89eab306-bbbd-11e8-b2ba-ac1f6b6435d0_red.png'),
  #  ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/5061784c-bbca-11e8-b2bc-ac1f6b6435d0_yellow.png',
  #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/89eab306-bbbd-11e8-b2ba-ac1f6b6435d0_yellow.png')]

  # ================================================================================
  # @ List format should be as follow for Augmentor

  # images=[[im_1, mask_1_a, mask_1_b],
  #         [im_2, mask_2_a, mask_2_b],
  #         ...,
  #         [im_n, mask_n_a, mask_n_b]]

  # y is label like [0,1,1,0,1,1] when you use 6 images which have binary class

  # ================================================================================
  B_imgs=list(paths_of_imgs[0])
  G_imgs=list(paths_of_imgs[1])
  R_imgs=list(paths_of_imgs[2])
  Y_imgs=list(paths_of_imgs[3])
  
  zipped_paths=list(zip(B_imgs,G_imgs,R_imgs,Y_imgs))
  # print("zipped_paths",zipped_paths)
  # [('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/5061784c-bbca-11e8-b2bc-ac1f6b6435d0_blue.png',
  #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/5061784c-bbca-11e8-b2bc-ac1f6b6435d0_green.png',
  #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/5061784c-bbca-11e8-b2bc-ac1f6b6435d0_red.png',
  #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/5061784c-bbca-11e8-b2bc-ac1f6b6435d0_yellow.png'),
  #  ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/89eab306-bbbd-11e8-b2ba-ac1f6b6435d0_blue.png',
  #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/89eab306-bbbd-11e8-b2ba-ac1f6b6435d0_green.png',
  #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/89eab306-bbbd-11e8-b2ba-ac1f6b6435d0_red.png',
  #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/89eab306-bbbd-11e8-b2ba-ac1f6b6435d0_yellow.png')]

  # ================================================================================
  # c labels_in_scalar: label values of images
  # [tensor([19, 12])]
  labels_in_scalar=np.array(data[1])
  # print("labels_in_scalar",labels_in_scalar)
  # ['0 21' '25 0']

  # ================================================================================
  # c bs_pa_tumor_d: batchsized paths of tumor dataset
  bs_pa_tumor_d=[zipped_paths,labels_in_scalar]
  # print("bs_pa_tumor_d",bs_pa_tumor_d)
  # [[('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/b327efac-bbbf-11e8-b2bb-ac1f6b6435d0_blue.png',
  #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/b327efac-bbbf-11e8-b2bb-ac1f6b6435d0_green.png',
  #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/b327efac-bbbf-11e8-b2bb-ac1f6b6435d0_red.png',
  #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/b327efac-bbbf-11e8-b2bb-ac1f6b6435d0_yellow.png'),
  #   ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/5684fad6-bb9e-11e8-b2b9-ac1f6b6435d0_blue.png',
  #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/5684fad6-bb9e-11e8-b2b9-ac1f6b6435d0_green.png',
  #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/5684fad6-bb9e-11e8-b2b9-ac1f6b6435d0_red.png',
  #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/5684fad6-bb9e-11e8-b2b9-ac1f6b6435d0_yellow.png')],
  #  array(['0 21','25 0'],dtype='<U4')]
  
  return bs_pa_tumor_d

# ================================================================================
def get_batch_vali_imgs(img_paths):
  all_images=[]
  for one_protein_vali in img_paths:

    loaded_b_img_vali=utils_image.load_img(one_protein_vali[0])
    loaded_g_img_vali=utils_image.load_img(one_protein_vali[1])
    loaded_r_img_vali=utils_image.load_img(one_protein_vali[2])
    loaded_y_img_vali=utils_image.load_img(one_protein_vali[3])
    # print("loaded_b_img_vali",loaded_b_img_vali.shape)
    # (512, 512)

    loaded_b_img_vali=utils_image.resize_img(loaded_b_img_vali,224,224)
    loaded_g_img_vali=utils_image.resize_img(loaded_g_img_vali,224,224)
    loaded_r_img_vali=utils_image.resize_img(loaded_r_img_vali,224,224)
    loaded_y_img_vali=utils_image.resize_img(loaded_y_img_vali,224,224)
    # print("loaded_b_img_vali",loaded_b_img_vali.shape)
    # (224, 224)

    all_images.append([loaded_b_img_vali,loaded_g_img_vali,loaded_r_img_vali,loaded_y_img_vali])
  
  all_images_vali_stacked=np.stack(all_images,axis=0)
  # print("all_images_vali_stacked",all_images_vali_stacked.shape)
  # (2, 4, 224, 224)

  return all_images_vali_stacked

def convert_char_to_num_for_DNA_Seq(PR_Seq):
  PR_Seq=PR_Seq.replace("A","1@").replace("B","2@").replace("C","3@").replace("D","4@").replace("E","5@").replace("F","6@").replace("G","7@").\
                replace("H","8@").replace("I","9@").replace("J","10@").replace("K","11@").replace("L","12@").replace("M","13@").replace("N","14@").\
                replace("O","15@").replace("P","16@").replace("Q","17@").replace("R","18@").replace("S","19@").replace("T","20@").replace("U","21@").\
                replace("V","22@").replace("W","23@").replace("X","24@").replace("Y","25@").replace("Z","26@")
  return PR_Seq

def process_PR_Seq(PR_Seq):
  # print("PR_Seq",PR_Seq.shape)
  # (612,)

  converted_DNA_seq=[]

  for i in range(PR_Seq.shape[0]):
    PR_Seq_num=convert_char_to_num_for_DNA_Seq(PR_Seq[i])[:-1]
    # print("PR_Seq_num",PR_Seq_num)
    # 3@3@20@3@1@1@1@20@3@1@3@20@3@20@20@20@7@7@3@1@1@3@7@1@3@3@3@3@20@3@7@20@3@3@3@1@1@20@1@1@7@7@1@20@1@7@7@7@7@7@7@3@1@1@3@20@1@1@1@7@7@1@1@7@3@25@3@20@1@20@20@1@7@1@20@1@3@1@7@7@1@7@3@1@7@1@20@7@1@20@1@3@1@7@20@1@20@20@1@7@1@1@7@1@3@1@20@7@7@1@7@20@20@7@3@3@1@7@7@1@1@7@1@20@7@7@1@1@1@3@3@1@1@1@1@1@20@7@1@20@1@7@7@7@7@7@1@1@20@20@7@7@1@7@7@20@20@20@20@1@20@3@1@1@1@7@20@1@1@18@1@3@1@7@20@1@20@7@1@20@3@1@7@18@20@1@3@3@3@1@20@1@7@1@1@1@20@3@20@1@20@7@7@1@3@1@20@1@1@1@7@3@20@7@20@1@7@7@20@1@3@1@7@20@1@20@20@1@1@20@1@7@7@1@3@3@20@1@3@1@3@3@20@7@20@3@1@1@3@1@20@1@1@20@20@7@7@1@1@7@1@1@1@20@3@20@7@20@20@7@1@3@20@3@1@7@3@20@20@7@7@20@20@7@3@1@3@20@20@20@1@1@1@20@20@20@25

    PR_Seq_num_li=PR_Seq_num.split("@")
    # print("PR_Seq_num_li",PR_Seq_num_li)
    # ['3', '3', '20', '3', '1', '1', '1', '20', '3', '1', '3', '20', '3', '20', '20', '20', '7', '7', '3', '1', '1', '3', '7', '1', '3', '3', '3', '3', '20', '3', '7', '20', '3', '3', '3', '1', '1', '20', '1', '1', '7', '7', '1', '20', '1', '7', '7', '7', '7', '7', '7', '3', '1', '1', '3', '20', '1', '1', '1', '7', '7', '1', '1', '7', '3', '25', '3', '20', '1', '20', '20', '1', '7', '1', '20', '1', '3', '1', '7', '7', '1', '7', '3', '1', '7', '1', '20', '7', '1', '20', '1', '3', '1', '7', '20', '1', '20', '20', '1', '7', '1', '1', '7', '1', '3', '1', '20', '7', '7', '1', '7', '20', '20', '7', '3', '3', '1', '7', '7', '1', '1', '7', '1', '20', '7', '7', '1', '1', '1', '3', '3', '1', '1', '1', '1', '1', '20', '7', '1', '20', '1', '7', '7', '7', '7', '7', '1', '1', '20', '20', '7', '7', '1', '7', '7', '20', '20', '20', '20', '1', '20', '3', '1', '1', '1', '7', '20', '1', '1', '18', '1', '3', '1', '7', '20', '1', '20', '7', '1', '20', '3', '1', '7', '18', '20', '1', '3', '3', '3', '1', '20', '1', '7', '1', '1', '1', '20', '3', '20', '1', '20', '7', '7', '1', '3', '1', '20', '1', '1', '1', '7', '3', '20', '7', '20', '1', '7', '7', '20', '1', '3', '1', '7', '20', '1', '20', '20', '1', '1', '20', '1', '7', '7', '1', '3', '3', '20', '1', '3', '1', '3', '3', '20', '7', '20', '3', '1', '1', '3', '1', '20', '1', '1', '20', '20', '7', '7', '1', '1', '7', '1', '1', '1', '20', '3', '20', '7', '20', '20', '7', '1', '3', '20', '3', '1', '7', '3', '20', '20', '7', '7', '20', '20', '7', '3', '1', '3', '20', '20', '20', '1', '1', '1', '20', '20', '20', '25']
    
    PR_Seq_num_li=list(map(int,PR_Seq_num_li))
    # print("PR_Seq_num_li",PR_Seq_num_li)
    # [3, 3, 20, 3, 1, 1, 1, 20, 3, 1, 3, 20, 3, 20, 20, 20, 7, 7, 3, 1, 1, 3, 7, 1, 3, 3, 3, 3, 20, 3, 7, 20, 3, 3, 3, 1, 1, 20, 1, 1, 7, 7, 1, 20, 1, 7, 7, 7, 7, 7, 7, 3, 1, 1, 3, 20, 1, 1, 1, 7, 7, 1, 1, 7, 3, 25, 3, 20, 1, 20, 20, 1, 7, 1, 20, 1, 3, 1, 7, 7, 1, 7, 3, 1, 7, 1, 20, 7, 1, 20, 1, 3, 1, 7, 20, 1, 20, 20, 1, 7, 1, 1, 7, 1, 3, 1, 20, 7, 7, 1, 7, 20, 20, 7, 3, 3, 1, 7, 7, 1, 1, 7, 1, 20, 7, 7, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 20, 7, 1, 20, 1, 7, 7, 7, 7, 7, 1, 1, 20, 20, 7, 7, 1, 7, 7, 20, 20, 20, 20, 1, 20, 3, 1, 1, 1, 7, 20, 1, 1, 18, 1, 3, 1, 7, 20, 1, 20, 7, 1, 20, 3, 1, 7, 18, 20, 1, 3, 3, 3, 1, 20, 1, 7, 1, 1, 1, 20, 3, 20, 1, 20, 7, 7, 1, 3, 1, 20, 1, 1, 1, 7, 3, 20, 7, 20, 1, 7, 7, 20, 1, 3, 1, 7, 20, 1, 20, 20, 1, 1, 20, 1, 7, 7, 1, 3, 3, 20, 1, 3, 1, 3, 3, 20, 7, 20, 3, 1, 1, 3, 1, 20, 1, 1, 20, 20, 7, 7, 1, 1, 7, 1, 1, 1, 20, 3, 20, 7, 20, 20, 7, 1, 3, 20, 3, 1, 7, 3, 20, 20, 7, 7, 20, 20, 7, 3, 1, 3, 20, 20, 20, 1, 1, 1, 20, 20, 20, 25]

    # norm_arr_1D=utils_common.normalize_1D_arr(PR_Seq_num_li)
    # print("norm_arr_1D",norm_arr_1D)
    # afaf

    converted_DNA_seq.append(PR_Seq_num_li)

  # print("converted_DNA_seq",converted_DNA_seq)
  # [[3, 3, 20, 3, 1, 1, 1, 20, 3, 1, 3, 20, 3, 20, 20, 20, 7, 7, 3, 1, 1, 3, 7, 1, 3, 3, 3, 3, 20, 3, 7, 20, 3, 3, 3, 1, 1, 20, 1, 1, 7, 7, 1, 20, 1, 7, 7, 7, 7, 7, 7, 3, 1, 1, 3, 20, 1, 1, 1, 7, 

  return converted_DNA_seq

def process_RT_Seq(RT_Seq):
  # print("RT_Seq",RT_Seq.shape)
  # (612,)

  converted_DNA_seq=[]

  for i in range(RT_Seq.shape[0]):
    RT_Seq_num=convert_char_to_num_for_DNA_Seq(RT_Seq[i])[:-1]
    # print("RT_Seq_num",RT_Seq_num)
    # 3@3@3@1@20@20@1@7@20@3@3@20@1@20@20@7@1@1@1@3@20@7@20@1@3@3@1@7@20@1@1@1@7@3@20@1@1@1@7@3@3@1@7@7@1@1@20@7@7@1@20@7@7@3@3@3@1@1@1@1@7@20@20@1@1@1@3@1@1@20@7@7@3@3@1@20@20@7@1@3@1@7@1@1@7@1@1@1@1@1@1@20@1@1@1@1@7@3@1@20@20@1@7@20@1@7@1@1@1@20@20@20@7@25@1@3@1@7@1@1@1@20@7@7@1@1@1@1@7@7@1@1@7@7@7@1@1@1@1@20@20@20@3@1@1@1@1@1@20@20@7@7@7@3@3@20@7@1@1@1@1@20@3@3@1@20@1@20@1@1@20@1@3@20@3@3@1@7@20@1@20@20@20@7@3@3@1@20@1@1@1@7@1@1@1@1@1@1@7@1@3@1@7@20@1@3@20@1@3@1@20@7@7@1@7@1@1@1@1@20@20@1@7@20@1@7@1@20@20@20@3@1@7@1@7@1@1@3@20@20@1@1@20@1@1@7@1@7@1@1@3@20@3@1@1@7@1@3@20@20@3@20@7@7@7@1@1@7@20@20@3@1@1@25@20@1@7@7@1@1@20@1@3@3@1@3@1@20@3@3@3@7@3@23@7@7@7@20@20@1@1@1@1@1@1@7@1@1@25@1@1@1@20@3@1@7@20@1@1@3@1@7@20@1@3@20@7@7@1@20@7@20@7@7@7@20@7@1@20@7@3@1@20@1@20@20@20@3@20@3@1@7@20@20@3@3@13@20@20@1@7@1@20@1@1@1@7@1@3@20@20@3@1@7@7@1@1@7@20@1@20@1@3@20@7@3@1@20@20@20@1@3@3@1@20@1@3@3@20@1@7@20@1@20@1@1@1@3@1@1@20@7@1@7@1@3@1@3@3@1@7@7@7@1@20@20@1@7@1@20@1@20@3@1@7@20@1@3@1@1@20@7@20@7@3@20@20@3@3@1@3@1@7@7@7@1@20@7@7@1@1@1@7@7@1@20@3@1@3@3@1@7@3@1@1@20@1@20@20@3@3@1@1@1@7@20@1@7@3@1@20@7@1@3@1@1@1@1@1@20@3@20@20@1@7@1@7@3@3@20@20@20@20@1@7@1@1@1@1@3@7@1@1@1@20@3@3@1@7@1@3@1@20@1@7@20@20@1@20@3@20@1@3@3@1@1@20@1@3@1@20@7@7@1@20@7@1@20@20@20@7@20@1@20@7@20@1@7@7@1@20@3@20@7@1@20@20@20@18@7@1@1@1@20@1@7@1@1@3@1@7@3@1@20@1@7@1@1@3@1@1@1@1@1@20@1@7@1@7@7@1@1@3@20@7@1@7@1@3@1@1@3@1@20@3@20@7@20@3@1@1@7@7@20@7@7@7@7@7@20@20@20@1@3@3@1@3@1@3@3@1@7@1@3@1@1@1@1@1@1@3@1@20@3@1@7@1@1@1@7@1@1@3@3@20@3@3@1@20@20@3@3@20@20@20@7@7@1@20@7@7@7@3@20@1@20@7@1@1@3@20@3@3@1@20@3@3@20@7@1@20@1@1@1@20@7@7@1@3@1@7@20@1@3@1@7@3@3@20@1@20@1@7@20@20@3@20@7@3@3@1@7@1@1@1@1@1@7@1@20@1@7@3@20@7@7@1@3@20@7@20@3@1@1@20@7@1@3@1@20@1@3@1@7@1@1@7@20@20@1@7@20@7@7@7@7@1@1@7@20@20@7@1@1@20@20@7@7@7@3@1@1@7@20@3@1@7@1@20@20@20@1@25@7@3@1@7@7@7@1@20@20@1@1@1@7@20@1@1@1@7@3@1@1@20@20@1@20@7@20@1@1@1@3@20@3@3@20@20@1@7@7@7@7@7@1@3@3@1@1@7@11@3@1@3@20@1@1@3@1@7@1@1@1@20@1@1@20@1@3@3@1@3@20@1@1@3@1@1@7@1@7@1@1@7@3@1@7@1@7@3@20@1@7@1@1@3@20@7@7@3@1@7@1@1@1@1@3@1@7@7@7@1@1@1@20@20@3@20@1@1@1@1@7@1@1@3@3@1@7@20@1@3@1@20@7@7@1@7@20@7@20@1@20@20@1@20@7@1@20@3@3@1@1@3@1@1@1@1@7@1@3@20@20@1@1@20@1@7@3@1@7@1@1@1@20@1@3@1@7@1@1@7@3@1@7@7@7@7@3@1@1@7@7@3

    RT_Seq_num_li=RT_Seq_num.split("@")
    # print("RT_Seq_num_li",RT_Seq_num_li)
    # ['3', '3', '3', '1', '20', '20', '1', '7', '20', '3', '3', '20', '1', '20', '20', '7', '1', '1', '1', '3', '20', '7', '20', '1', '3', '3', '1', '7', '20', '1', '1', '1', '7', '3', '20', '1', '1', '1', '7', '3', '3', '1', '7', '7', '1', '1', '20', '7', '7', '1', '20', '7', '7', '3', '3', '3', '1', '1', '1', '1', '7', '20', '20', '1', '1', '1', '3', '1', '1', '20', '7', '7', '3', '3', '1', '20', '20', '7', '1', '3', '1', '7', '1', '1', '7', '1', '1', '1', '1', '1', '1', '20', '1', '1', '1', '1', '7', '3', '1', '20', '20', '1', '7', '20', '1', '7', '1', '1', '1', '20', '20', '20', '7', '25', '1', '3', '1', '7', '1', '1', '1', '20', '7', '7', '1', '1', '1', '1', '7', '7', '1', '1', '7', '7', '7', '1', '1', '1', '1', '20', '20', '20', '3', '1', '1', '1', '1', '1', '20', '20', '7', '7', '7', '3', '3', '20', '7', '1', '1', '1', '1', '20', '3', '3', '1', '20', '1', '20', '1', '1', '20', '1', '3', '20', '3', '3', '1', '7', '20', '1', '20', '20', '20', '7', '3', '3', '1', '20', '1', '1', '1', '7', '1', '1', '1', '1', '1', '1', '7', '1', '3', '1', '7', '20', '1', '3', '20', '1', '3', '1', '20', '7', '7', '1', '7', '1', '1', '1', '1', '20', '20', '1', '7', '20', '1', '7', '1', '20', '20', '20', '3', '1', '7', '1', '7', '1', '1', '3', '20', '20', '1', '1', '20', '1', '1', '7', '1', '7', '1', '1', '3', '20', '3', '1', '1', '7', '1', '3', '20', '20', '3', '20', '7', '7', '7', '1', '1', '7', '20', '20', '3', '1', '1', '25', '20', '1', '7', '7', '1', '1', '20', '1', '3', '3', '1', '3', '1', '20', '3', '3', '3', '7', '3', '23', '7', '7', '7', '20', '20', '1', '1', '1', '1', '1', '1', '7', '1', '1', '25', '1', '1', '1', '20', '3', '1', '7', '20', '1', '1', '3', '1', '7', '20', '1', '3', '20', '7', '7', '1', '20', '7', '20', '7', '7', '7', '20', '7', '1', '20', '7', '3', '1', '20', '1', '20', '20', '20', '3', '20', '3', '1', '7', '20', '20', '3', '3', '13', '20', '20', '1', '7', '1', '20', '1', '1', '1', '7', '1', '3', '20', '20', '3', '1', '7', '7', '1', '1', '7', '20', '1', '20', '1', '3', '20', '7', '3', '1', '20', '20', '20', '1', '3', '3', '1', '20', '1', '3', '3', '20', '1', '7', '20', '1', '20', '1', '1', '1', '3', '1', '1', '20', '7', '1', '7', '1', '3', '1', '3', '3', '1', '7', '7', '7', '1', '20', '20', '1', '7', '1', '20', '1', '20', '3', '1', '7', '20', '1', '3', '1', '1', '20', '7', '20', '7', '3', '20', '20', '3', '3', '1', '3', '1', '7', '7', '7', '1', '20', '7', '7', '1', '1', '1', '7', '7', '1', '20', '3', '1', '3', '3', '1', '7', '3', '1', '1', '20', '1', '20', '20', '3', '3', '1', '1', '1', '7', '20', '1', '7', '3', '1', '20', '7', '1', '3', '1', '1', '1', '1', '1', '20', '3', '20', '20', '1', '7', '1', '7', '3', '3', '20', '20', '20', '20', '1', '7', '1', '1', '1', '1', '3', '7', '1', '1', '1', '20', '3', '3', '1', '7', '1', '3', '1', '20', '1', '7', '20', '20', '1', '20', '3', '20', '1', '3', '3', '1', '1', '20', '1', '3', '1', '20', '7', '7', '1', '20', '7', '1', '20', '20', '20', '7', '20', '1', '20', '7', '20', '1', '7', '7', '1', '20', '3', '20', '7', '1', '20', '20', '20', '18', '7', '1', '1', '1', '20', '1', '7', '1', '1', '3', '1', '7', '3', '1', '20', '1', '7', '1', '1', '3', '1', '1', '1', '1', '1', '20', '1', '7', '1', '7', '7', '1', '1', '3', '20', '7', '1', '7', '1', '3', '1', '1', '3', '1', '20', '3', '20', '7', '20', '3', '1', '1', '7', '7', '20', '7', '7', '7', '7', '7', '20', '20', '20', '1', '3', '3', '1', '3', '1', '3', '3', '1', '7', '1', '3', '1', '1', '1', '1', '1', '1', '3', '1', '20', '3', '1', '7', '1', '1', '1', '7', '1', '1', '3', '3', '20', '3', '3', '1', '20', '20', '3', '3', '20', '20', '20', '7', '7', '1', '20', '7', '7', '7', '3', '20', '1', '20', '7', '1', '1', '3', '20', '3', '3', '1', '20', '3', '3', '20', '7', '1', '20', '1', '1', '1', '20', '7', '7', '1', '3', '1', '7', '20', '1', '3', '1', '7', '3', '3', '20', '1', '20', '1', '7', '20', '20', '3', '20', '7', '3', '3', '1', '7', '1', '1', '1', '1', '1', '7', '1', '20', '1', '7', '3', '20', '7', '7', '1', '3', '20', '7', '20', '3', '1', '1', '20', '7', '1', '3', '1', '20', '1', '3', '1', '7', '1', '1', '7', '20', '20', '1', '7', '20', '7', '7', '7', '7', '1', '1', '7', '20', '20', '7', '1', '1', '20', '20', '7', '7', '7', '3', '1', '1', '7', '20', '3', '1', '7', '1', '20', '20', '20', '1', '25', '7', '3', '1', '7', '7', '7', '1', '20', '20', '1', '1', '1', '7', '20', '1', '1', '1', '7', '3', '1', '1', '20', '20', '1', '20', '7', '20', '1', '1', '1', '3', '20', '3', '3', '20', '20', '1', '7', '7', '7', '7', '7', '1', '3', '3', '1', '1', '7', '11', '3', '1', '3', '20', '1', '1', '3', '1', '7', '1', '1', '1', '20', '1', '1', '20', '1', '3', '3', '1', '3', '20', '1', '1', '3', '1', '1', '7', '1', '7', '1', '1', '7', '3', '1', '7', '1', '7', '3', '20', '1', '7', '1', '1', '3', '20', '7', '7', '3', '1', '7', '1', '1', '1', '1', '3', '1', '7', '7', '7', '1', '1', '1', '20', '20', '3', '20', '1', '1', '1', '1', '7', '1', '1', '3', '3', '1', '7', '20', '1', '3', '1', '20', '7', '7', '1', '7', '20', '7', '20', '1', '20', '20', '1', '20', '7', '1', '20', '3', '3', '1', '1', '3', '1', '1', '1', '1', '7', '1', '3', '20', '20', '1', '1', '20', '1', '7', '3', '1', '7', '1', '1', '1', '20', '1', '3', '1', '7', '1', '1', '7', '3', '1', '7', '7', '7', '7', '3', '1', '1', '7', '7', '3']
    
    RT_Seq_num_li=list(map(int,RT_Seq_num_li))
    # print("RT_Seq_num_li",RT_Seq_num_li)
    # [3, 3, 3, 1, 20, 20, 1, 7, 20, 3, 3, 20, 1, 20, 20, 7, 1, 1, 1, 3, 20, 7, 20, 1, 3, 3, 1, 7, 20, 1, 1, 1, 7, 3, 20, 1, 1, 1, 7, 3, 3, 1, 7, 7, 1, 1, 20, 7, 7, 1, 20, 7, 7, 3, 3, 3, 1, 1, 1, 1, 7, 20, 20, 1, 1, 1, 3, 1, 1, 20, 7, 7, 3, 3, 1, 20, 20, 7, 1, 3, 1, 7, 1, 1, 7, 1, 1, 1, 1, 1, 1, 20, 1, 1, 1, 1, 7, 3, 1, 20, 20, 1, 7, 20, 1, 7, 1, 1, 1, 20, 20, 20, 7, 25, 1, 3, 1, 7, 1, 1, 1, 20, 7, 7, 1, 1, 1, 1, 7, 7, 1, 1, 7, 7, 7, 1, 1, 1, 1, 20, 20, 20, 3, 1, 1, 1, 1, 1, 20, 20, 7, 7, 7, 3, 3, 20, 7, 1, 1, 1, 1, 20, 3, 3, 1, 20, 1, 20, 1, 1, 20, 1, 3, 20, 3, 3, 1, 7, 20, 1, 20, 20, 20, 7, 3, 3, 1, 20, 1, 1, 1, 7, 1, 1, 1, 1, 1, 1, 7, 1, 3, 1, 7, 20, 1, 3, 20, 1, 3, 1, 20, 7, 7, 1, 7, 1, 1, 1, 1, 20, 20, 1, 7, 20, 1, 7, 1, 20, 20, 20, 3, 1, 7, 1, 7, 1, 1, 3, 20, 20, 1, 1, 20, 1, 1, 7, 1, 7, 1, 1, 3, 20, 3, 1, 1, 7, 1, 3, 20, 20, 3, 20, 7, 7, 7, 1, 1, 7, 20, 20, 3, 1, 1, 25, 20, 1, 7, 7, 1, 1, 20, 1, 3, 3, 1, 3, 1, 20, 3, 3, 3, 7, 3, 23, 7, 7, 7, 20, 20, 1, 1, 1, 1, 1, 1, 7, 1, 1, 25, 1, 1, 1, 20, 3, 1, 7, 20, 1, 1, 3, 1, 7, 20, 1, 3, 20, 7, 7, 1, 20, 7, 20, 7, 7, 7, 20, 7, 1, 20, 7, 3, 1, 20, 1, 20, 20, 20, 3, 20, 3, 1, 7, 20, 20, 3, 3, 13, 20, 20, 1, 7, 1, 20, 1, 1, 1, 7, 1, 3, 20, 20, 3, 1, 7, 7, 1, 1, 7, 20, 1, 20, 1, 3, 20, 7, 3, 1, 20, 20, 20, 1, 3, 3, 1, 20, 1, 3, 3, 20, 1, 7, 20, 1, 20, 1, 1, 1, 3, 1, 1, 20, 7, 1, 7, 1, 3, 1, 3, 3, 1, 7, 7, 7, 1, 20, 20, 1, 7, 1, 20, 1, 20, 3, 1, 7, 20, 1, 3, 1, 1, 20, 7, 20, 7, 3, 20, 20, 3, 3, 1, 3, 1, 7, 7, 7, 1, 20, 7, 7, 1, 1, 1, 7, 7, 1, 20, 3, 1, 3, 3, 1, 7, 3, 1, 1, 20, 1, 20, 20, 3, 3, 1, 1, 1, 7, 20, 1, 7, 3, 1, 20, 7, 1, 3, 1, 1, 1, 1, 1, 20, 3, 20, 20, 1, 7, 1, 7, 3, 3, 20, 20, 20, 20, 1, 7, 1, 1, 1, 1, 3, 7, 1, 1, 1, 20, 3, 3, 1, 7, 1, 3, 1, 20, 1, 7, 20, 20, 1, 20, 3, 20, 1, 3, 3, 1, 1, 20, 1, 3, 1, 20, 7, 7, 1, 20, 7, 1, 20, 20, 20, 7, 20, 1, 20, 7, 20, 1, 7, 7, 1, 20, 3, 20, 7, 1, 20, 20, 20, 18, 7, 1, 1, 1, 20, 1, 7, 1, 1, 3, 1, 7, 3, 1, 20, 1, 7, 1, 1, 3, 1, 1, 1, 1, 1, 20, 1, 7, 1, 7, 7, 1, 1, 3, 20, 7, 1, 7, 1, 3, 1, 1, 3, 1, 20, 3, 20, 7, 20, 3, 1, 1, 7, 7, 20, 7, 7, 7, 7, 7, 20, 20, 20, 1, 3, 3, 1, 3, 1, 3, 3, 1, 7, 1, 3, 1, 1, 1, 1, 1, 1, 3, 1, 20, 3, 1, 7, 1, 1, 1, 7, 1, 1, 3, 3, 20, 3, 3, 1, 20, 20, 3, 3, 20, 20, 20, 7, 7, 1, 20, 7, 7, 7, 3, 20, 1, 20, 7, 1, 1, 3, 20, 3, 3, 1, 20, 3, 3, 20, 7, 1, 20, 1, 1, 1, 20, 7, 7, 1, 3, 1, 7, 20, 1, 3, 1, 7, 3, 3, 20, 1, 20, 1, 7, 20, 20, 3, 20, 7, 3, 3, 1, 7, 1, 1, 1, 1, 1, 7, 1, 20, 1, 7, 3, 20, 7, 7, 1, 3, 20, 7, 20, 3, 1, 1, 20, 7, 1, 3, 1, 20, 1, 3, 1, 7, 1, 1, 7, 20, 20, 1, 7, 20, 7, 7, 7, 7, 1, 1, 7, 20, 20, 7, 1, 1, 20, 20, 7, 7, 7, 3, 1, 1, 7, 20, 3, 1, 7, 1, 20, 20, 20, 1, 25, 7, 3, 1, 7, 7, 7, 1, 20, 20, 1, 1, 1, 7, 20, 1, 1, 1, 7, 3, 1, 1, 20, 20, 1, 20, 7, 20, 1, 1, 1, 3, 20, 3, 3, 20, 20, 1, 7, 7, 7, 7, 7, 1, 3, 3, 1, 1, 7, 11, 3, 1, 3, 20, 1, 1, 3, 1, 7, 1, 1, 1, 20, 1, 1, 20, 1, 3, 3, 1, 3, 20, 1, 1, 3, 1, 1, 7, 1, 7, 1, 1, 7, 3, 1, 7, 1, 7, 3, 20, 1, 7, 1, 1, 3, 20, 7, 7, 3, 1, 7, 1, 1, 1, 1, 3, 1, 7, 7, 7, 1, 1, 1, 20, 20, 3, 20, 1, 1, 1, 1, 7, 1, 1, 3, 3, 1, 7, 20, 1, 3, 1, 20, 7, 7, 1, 7, 20, 7, 20, 1, 20, 20, 1, 20, 7, 1, 20, 3, 3, 1, 1, 3, 1, 1, 1, 1, 7, 1, 3, 20, 20, 1, 1, 20, 1, 7, 3, 1, 7, 1, 1, 1, 20, 1, 3, 1, 7, 1, 1, 7, 3, 1, 7, 7, 7, 7, 3, 1, 1, 7, 7, 3]

    # norm_arr_1D=utils_common.normalize_1D_arr(RT_Seq_num_li)
    # print("norm_arr_1D",norm_arr_1D)
    # afaf

    converted_DNA_seq.append(RT_Seq_num_li)

  # print("converted_DNA_seq",converted_DNA_seq)
  # [[3, 3, 3, 1, 20, 20, 1, 7, 20, 3, 3, 20, 1, 20, 20, 7, 1, 1, 1, 3, 20, 7, 20, 1, 3, 3, 1, 7, 20, 1, 1, 1, 7, 3, 20, 1, 1, 1, 7, 3, 3, 1, 7, 7, 1, 1, 20, 7, 7, 1, 20, 7, 7, 3, 3, 3, 1, 1, 1, 

  return converted_DNA_seq
  
def one_hot_PR_Seq(PR_Seq_converted):
  min_val=np.min(np.min(PR_Seq_converted))
  # print("min_val",min_val)
  # 1
  max_val=np.max(np.max(PR_Seq_converted))
  # print("max_val",max_val)
  # 20


  faf
  for one_PR in PR_Seq_converted:
    print("one_PR",one_PR)
    afaf

def get_one_dummy_data_for_test():

  # dummy_data_for_test=
  one_PR_Seq=np.array([3.,3.,20.,3.,1.,1.,1.,20.,3.,1.,3.,20.,3.,20.,20.,20.,7.,7.,3.,1.,1.,3.,7.,1.,3.,3.,3.,3.,20.,3.,7.,20.,3.,3.,3.,1.,1.,20.,1.,1.,7.,7.,1.,20.,1.,7.,7.,7.,7.,7.,7.,3.,1.,1.,3.,20.,1.,1.,1.,7.,7.,1.,1.,7.,3.,25.,3.,20.,1.,20.,20.,1.,7.,1.,20.,1.,3.,1.,7.,7.,1.,7.,3.,1.,7.,1.,20.,7.,1.,20.,1.,3.,1.,7.,20.,1.,20.,20.,1.,7.,1.,1.,7.,1.,3.,1.,20.,7.,7.,1.,7.,20.,20.,7.,3.,3.,1.,7.,7.,1.,1.,7.,1.,20.,7.,7.,1.,1.,1.,3.,3.,1.,1.,1.,1.,1.,20.,7.,1.,20.,1.,7.,7.,7.,7.,7.,1.,1.,20.,20.,7.,7.,1.,7.,7.,20.,20.,20.,20.,1.,20.,3.,1.,1.,1.,7.,20.,1.,1.,18.,1.,3.,1.,7.,20.,1.,20.,7.,1.,20.,3.,1.,7.,18.,20.,1.,3.,3.,3.,1.,20.,1.,7.,1.,1.,1.,20.,3.,20.,1.,20.,7.,7.,1.,3.,1.,20.,1.,1.,1.,7.,3.,20.,7.,20.,1.,7.,7.,20.,1.,3.,1.,7.,20.,1.,20.,20.,1.,1.,20.,1.,7.,7.,1.,3.,3.,20.,1.,3.,1.,3.,3.,20.,7.,20.,3.,1.,1.,3.,1.,20.,1.,1.,20.,20.,7.,7.,1.,1.,7.,1.,1.,1.,20.,3.,20.,7.,20.,20.,7.,1.,3.,20.,3.,1.,7.,3.,20.,20.,7.,7.,20.,20.,7.,3.,1.,3.,20.,20.,20.,1.,1.,1.,20.,20.,20.,25.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
  # print("one_PR_Seq",one_PR_Seq.shape)
  # (386,)

  one_RT_Seq=np.array([3.,3.,3.,1.,20.,20.,1.,7.,20.,3.,3.,20.,1.,20.,20.,7.,1.,1.,1.,3.,20.,7.,20.,1.,3.,3.,1.,7.,20.,1.,1.,1.,7.,3.,20.,1.,1.,1.,7.,3.,3.,1.,7.,7.,1.,1.,20.,7.,7.,1.,20.,7.,7.,3.,3.,3.,1.,1.,1.,1.,7.,20.,20.,1.,1.,1.,3.,1.,1.,20.,7.,7.,3.,3.,1.,20.,20.,7.,1.,3.,1.,7.,1.,1.,7.,1.,1.,1.,1.,1.,1.,20.,1.,1.,1.,1.,7.,3.,1.,20.,20.,1.,7.,20.,1.,7.,1.,1.,1.,20.,20.,20.,7.,25.,1.,3.,1.,7.,1.,1.,1.,20.,7.,7.,1.,1.,1.,1.,7.,7.,1.,1.,7.,7.,7.,1.,1.,1.,1.,20.,20.,20.,3.,1.,1.,1.,1.,1.,20.,20.,7.,7.,7.,3.,3.,20.,7.,1.,1.,1.,1.,20.,3.,3.,1.,20.,1.,20.,1.,1.,20.,1.,3.,20.,3.,3.,1.,7.,20.,1.,20.,20.,20.,7.,3.,3.,1.,20.,1.,1.,1.,7.,1.,1.,1.,1.,1.,1.,7.,1.,3.,1.,7.,20.,1.,3.,20.,1.,3.,1.,20.,7.,7.,1.,7.,1.,1.,1.,1.,20.,20.,1.,7.,20.,1.,7.,1.,20.,20.,20.,3.,1.,7.,1.,7.,1.,1.,3.,20.,20.,1.,1.,20.,1.,1.,7.,1.,7.,1.,1.,3.,20.,3.,1.,1.,7.,1.,3.,20.,20.,3.,20.,7.,7.,7.,1.,1.,7.,20.,20.,3.,1.,1.,25.,20.,1.,7.,7.,1.,1.,20.,1.,3.,3.,1.,3.,1.,20.,3.,3.,3.,7.,3.,23.,7.,7.,7.,20.,20.,1.,1.,1.,1.,1.,1.,7.,1.,1.,25.,1.,1.,1.,20.,3.,1.,7.,20.,1.,1.,3.,1.,7.,20.,1.,3.,20.,7.,7.,1.,20.,7.,20.,7.,7.,7.,20.,7.,1.,20.,7.,3.,1.,20.,1.,20.,20.,20.,3.,20.,3.,1.,7.,20.,20.,3.,3.,13.,20.,20.,1.,7.,1.,20.,1.,1.,1.,7.,1.,3.,20.,20.,3.,1.,7.,7.,1.,1.,7.,20.,1.,20.,1.,3.,20.,7.,3.,1.,20.,20.,20.,1.,3.,3.,1.,20.,1.,3.,3.,20.,1.,7.,20.,1.,20.,1.,1.,1.,3.,1.,1.,20.,7.,1.,7.,1.,3.,1.,3.,3.,1.,7.,7.,7.,1.,20.,20.,1.,7.,1.,20.,1.,20.,3.,1.,7.,20.,1.,3.,1.,1.,20.,7.,20.,7.,3.,20.,20.,3.,3.,1.,3.,1.,7.,7.,7.,1.,20.,7.,7.,1.,1.,1.,7.,7.,1.,20.,3.,1.,3.,3.,1.,7.,3.,1.,1.,20.,1.,20.,20.,3.,3.,1.,1.,1.,7.,20.,1.,7.,3.,1.,20.,7.,1.,3.,1.,1.,1.,1.,1.,20.,3.,20.,20.,1.,7.,1.,7.,3.,3.,20.,20.,20.,20.,1.,7.,1.,1.,1.,1.,3.,7.,1.,1.,1.,20.,3.,3.,1.,7.,1.,3.,1.,20.,1.,7.,20.,20.,1.,20.,3.,20.,1.,3.,3.,1.,1.,20.,1.,3.,1.,20.,7.,7.,1.,20.,7.,1.,20.,20.,20.,7.,20.,1.,20.,7.,20.,1.,7.,7.,1.,20.,3.,20.,7.,1.,20.,20.,20.,18.,7.,1.,1.,1.,20.,1.,7.,1.,1.,3.,1.,7.,3.,1.,20.,1.,7.,1.,1.,3.,1.,1.,1.,1.,1.,20.,1.,7.,1.,7.,7.,1.,1.,3.,20.,7.,1.,7.,1.,3.,1.,1.,3.,1.,20.,3.,20.,7.,20.,3.,1.,1.,7.,7.,20.,7.,7.,7.,7.,7.,20.,20.,20.,1.,3.,3.,1.,3.,1.,3.,3.,1.,7.,1.,3.,1.,1.,1.,1.,1.,1.,3.,1.,20.,3.,1.,7.,1.,1.,1.,7.,1.,1.,3.,3.,20.,3.,3.,1.,20.,20.,3.,3.,20.,20.,20.,7.,7.,1.,20.,7.,7.,7.,3.,20.,1.,20.,7.,1.,1.,3.,20.,3.,3.,1.,20.,3.,3.,20.,7.,1.,20.,1.,1.,1.,20.,7.,7.,1.,3.,1.,7.,20.,1.,3.,1.,7.,3.,3.,20.,1.,20.,1.,7.,20.,20.,3.,20.,7.,3.,3.,1.,7.,1.,1.,1.,1.,1.,7.,1.,20.,1.,7.,3.,20.,7.,7.,1.,3.,20.,7.,20.,3.,1.,1.,20.,7.,1.,3.,1.,20.,1.,3.,1.,7.,1.,1.,7.,20.,20.,1.,7.,20.,7.,7.,7.,7.,1.,1.,7.,20.,20.,7.,1.,1.,20.,20.,7.,7.,7.,3.,1.,1.,7.,20.,3.,1.,7.,1.,20.,20.,20.,1.,25.,7.,3.,1.,7.,7.,7.,1.,20.,20.,1.,1.,1.,7.,20.,1.,1.,1.,7.,3.,1.,1.,20.,20.,1.,20.,7.,20.,1.,1.,1.,3.,20.,3.,3.,20.,20.,1.,7.,7.,7.,7.,7.,1.,3.,3.,1.,1.,7.,11.,3.,1.,3.,20.,1.,1.,3.,1.,7.,1.,1.,1.,20.,1.,1.,20.,1.,3.,3.,1.,3.,20.,1.,1.,3.,1.,1.,7.,1.,7.,1.,1.,7.,3.,1.,7.,1.,7.,3.,20.,1.,7.,1.,1.,3.,20.,7.,7.,3.,1.,7.,1.,1.,1.,1.,3.,1.,7.,7.,7.,1.,1.,1.,20.,20.,3.,20.,1.,1.,1.,1.,7.,1.,1.,3.,3.,1.,7.,20.,1.,3.,1.,20.,7.,7.,1.,7.,20.,7.,20.,1.,20.,20.,1.,20.,7.,1.,20.,3.,3.,1.,1.,3.,1.,1.,1.,1.,7.,1.,3.,20.,20.,1.,1.,20.,1.,7.,3.,1.,7.,1.,1.,1.,20.,1.,3.,1.,7.,1.,1.,7.,3.,1.,7.,7.,7.,7.,3.,1.,1.,7.,7.,3.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
  # print("one_RT_Seq",one_RT_Seq.shape)
  # (1822,)

  one_VL=np.array([4.3])

  one_CD4=np.array([145.0])

  one_dummy_data_for_test=np.hstack((one_PR_Seq,one_RT_Seq,one_VL,one_CD4))
  # print("one_dummy_data_for_test",one_dummy_data_for_test.shape)
  # (2210,)

  return one_dummy_data_for_test

def length_match_for_PR_Seq(PR_Seq):
  # print("PR_Seq",PR_Seq)
  # print("PR_Seq",PR_Seq.shape)
  # (920,)
  # print("PR_Seq.str.len().max()",PR_Seq.str.len())
  # 0      297
  # 1      297
  # print("PR_Seq.str.len().max()",PR_Seq.str.len().max())
  # 297

  PR_Seq_li=PR_Seq.tolist()
  # print("PR_Seq_li",PR_Seq_li)
  # ['CCTCAAATCACTCTTTGGCAACGACCCCTCGTCCCAATAAGGATAGGGGGGCAACTAAAGGAAGCYCTATTAGATACAGGAGCAGATGATACAGTATTAGAAGACATGGAGTTGCCAGGAAGATGGAAACCAAAAATGATAGGGGGAATTGGAGGTTTTATCAAAGTAARACAGTATGATCAGRTACCCATAGAAATCTATGGACATAAAGCTGTAGGTACAGTATTAATAGGACCTACACCTGTCAACATAATTGGAAGAAATCTGTTGACTCAGCTTGGTTGCACTTTAAATTTY', 
  
  # print("PR_Seq_li",len(PR_Seq_li))
  # 920

  new_PR_Seq_li=[]
  for one_PR_seq in PR_Seq_li:
    # print("one_PR_seq",one_PR_seq)
    # CCTCAAATCACTCTTTGGCAACGACCCCTCGTCCCAATAAGGATAGGGGGGCAACTAAAGGAAGCYCTATTAGATACAGGAGCAGATGATACAGTATTAGAAGACATGGAGTTGCCAGGAAGATGGAAACCAAAAATGATAGGGGGAATTGGAGGTTTTATCAAAGTAARACAGTATGATCAGRTACCCATAGAAATCTATGGACATAAAGCTGTAGGTACAGTATTAATAGGACCTACACCTGTCAACATAATTGGAAGAAATCTGTTGACTCAGCTTGGTTGCACTTTAAATTTY

    if len(one_PR_seq)!=297:
      lack_num=297-len(one_PR_seq)
      appended_0s="0"*lack_num
      # print("appended_0s",appended_0s)
      # appended_0s 000

      one_PR_seq=one_PR_seq+appended_0s
    else:
      pass
    
    new_PR_Seq_li.append(one_PR_seq)

  # print("new_PR_Seq_li",new_PR_Seq_li)
  # ['CCTCAAATCACTCTTTGGCAACGACCCCTCGTCCCAATAAGGATAGGGGGGCAACTAAAGGAAGCYCTATTAGATACAGGAGCAGATGATACAGTATTAGAAGACATGGAGTTGCCAGGAAGATGGAAACCAAAAATGATAGGGGGAATTGGAGGTTTTATCAAAGTAARACAGTATGATCAGRTACCCATAGAAATCTATGGACATAAAGCTGTAGGTACAGTATTAATAGGACCTACACCTGTCAACATAATTGGAAGAAATCTGTTGACTCAGCTTGGTTGCACTTTAAATTTY', 


  return new_PR_Seq_li

def length_match_for_RT_Seq(RT_Seq):

  # print("RT_Seq",RT_Seq.shape)
  # (920,)
  # print("RT_Seq.str.len().max()",RT_Seq.str.len())
  # 0      1005
  # 1      909 
  # print("RT_Seq.str.len().max()",RT_Seq.str.len().max())
  # 1482

  # ================================================================================
  RT_Seq_li=RT_Seq.tolist()
  # print("RT_Seq_li",RT_Seq_li)
  # ['CCCATTAGTCCTATTGAAACTGTACCAGTAAAGCTAAAGCCAGGAATGGATGGCCCAAAAGTTAAACAATGGCCATTGACAGAAGAAAAAATAAAAGCATTAGTAGAAATTTGYACAGAAATGGAAAAGGAAGGGAAAATTTCAAAAATTGGGCCTGAAAATCCATATAATACTCCAGTATTTGCCATAAAGAAAAAAGACAGTACTACATGGAGAAAATTAGTAGATTTCAGAGAACTTAATAAGAGAACTCAAGACTTCTGGGAAGTTCAAYTAGGAATACCACATCCCGCWGGGTTAAAAAAGAAYAAATCAGTAACAGTACTGGATGTGGGTGATGCATATTTCTCAGTTCCMTTAGATAAAGACTTCAGGAAGTATACTGCATTTACCATACCTAGTATAAACAATGAGACACCAGGGATTAGATATCAGTACAATGTGCTTCCACAGGGATGGAAAGGATCACCAGCAATATTCCAAAGTAGCATGACAAAAATCTTAGAGCCTTTTAGAAAACGAAATCCAGACATAGTTATCTACCAATACATGGATGATTTGTATGTAGGATCTGATTTRGAAATAGAACAGCATAGAACAAAAATAGAGGAACTGAGACAACATCTGTCAAGGTGGGGGTTTACCACACCAGACAAAAAACATCAGAAAGAACCTCCATTCCTTTGGATGGGCTATGAACTCCATCCTGATAAATGGACAGTACAGCCTATAGTTCTGCCAGAAAAAGATAGCTGGACTGTCAATGACATACAGAAGTTAGTGGGGAAGTTGAATTGGGCAAGTCAGATTTAYGCAGGGATTAAAGTAAAGCAATTATGTAAACTCCTTAGGGGGACCAAGKCACTAACAGAAATAATACCACTAACAAGAGAAGCAGAGCTAGAACTGGCAGAAAACAGGGAAATTCTAAAAGAACCAGTACATGGAGTGTATTATGATCCAACAAAAGACTTAATAGCAGAAATACAGAAGCAGGGGCAAGGC', 
  
  # print("PR_Seq_li",len(PR_Seq_li))
  # 920

  new_RT_Seq_li=[]
  for one_RT_seq in RT_Seq_li:
    # print("one_RT_seq",one_RT_seq)
    # afaf
    # CCCATTAGTCCTATTGAAACTGTACCAGTAAAGCTAAAGCCAGGAATGGATGGCCCAAAAGTTAAACAATGGCCATTGACAGAAGAAAAAATAAAAGCATTAGTAGAAATTTGYACAGAAATGGAAAAGGAAGGGAAAATTTCAAAAATTGGGCCTGAAAATCCATATAATACTCCAGTATTTGCCATAAAGAAAAAAGACAGTACTACATGGAGAAAATTAGTAGATTTCAGAGAACTTAATAAGAGAACTCAAGACTTCTGGGAAGTTCAAYTAGGAATACCACATCCCGCWGGGTTAAAAAAGAAYAAATCAGTAACAGTACTGGATGTGGGTGATGCATATTTCTCAGTTCCMTTAGATAAAGACTTCAGGAAGTATACTGCATTTACCATACCTAGTATAAACAATGAGACACCAGGGATTAGATATCAGTACAATGTGCTTCCACAGGGATGGAAAGGATCACCAGCAATATTCCAAAGTAGCATGACAAAAATCTTAGAGCCTTTTAGAAAACGAAATCCAGACATAGTTATCTACCAATACATGGATGATTTGTATGTAGGATCTGATTTRGAAATAGAACAGCATAGAACAAAAATAGAGGAACTGAGACAACATCTGTCAAGGTGGGGGTTTACCACACCAGACAAAAAACATCAGAAAGAACCTCCATTCCTTTGGATGGGCTATGAACTCCATCCTGATAAATGGACAGTACAGCCTATAGTTCTGCCAGAAAAAGATAGCTGGACTGTCAATGACATACAGAAGTTAGTGGGGAAGTTGAATTGGGCAAGTCAGATTTAYGCAGGGATTAAAGTAAAGCAATTATGTAAACTCCTTAGGGGGACCAAGKCACTAACAGAAATAATACCACTAACAAGAGAAGCAGAGCTAGAACTGGCAGAAAACAGGGAAATTCTAAAAGAACCAGTACATGGAGTGTATTATGATCCAACAAAAGACTTAATAGCAGAAATACAGAAGCAGGGGCAAGGC

    if len(one_RT_seq)!=1482:
      lack_num=1482-len(one_RT_seq)
      appended_0s="0"*lack_num
      # print("appended_0s",appended_0s)
      # appended_0s 000

      one_RT_seq=one_RT_seq+appended_0s
    else:
      pass
    
    new_RT_Seq_li.append(one_RT_seq)

  # print("new_RT_Seq_li",new_RT_Seq_li)
  # ['CCCATTAGTCCTATTGAAACTGTACCAGTAAAGCTAAAGCCAGGAATGGATGGCCCAAAAGTTAAACAATGGCCATTGACAGAAGAAAAAATAAAAGCATTAGTAGAAATTTGYACAGAAATGGAAAAGGAAGGGAAAATTTCAAAAATTGGGCCTGAAAATCCATATAATACTCCAGTATTTGCCATAAAGAAAAAAGACAGTACTACATGGAGAAAATTAGTAGATTTCAGAGAACTTAATAAGAGAACTCAAGACTTCTGGGAAGTTCAAYTAGGAATACCACATCCCGCWGGGTTAAAAAAGAAYAAATCAGTAACAGTACTGGATGTGGGTGATGCATATTTCTCAGTTCCMTTAGATAAAGACTTCAGGAAGTATACTGCATTTACCATACCTAGTATAAACAATGAGACACCAGGGATTAGATATCAGTACAATGTGCTTCCACAGGGATGGAAAGGATCACCAGCAATATTCCAAAGTAGCATGACAAAAATCTTAGAGCCTTTTAGAAAACGAAATCCAGACATAGTTATCTACCAATACATGGATGATTTGTATGTAGGATCTGATTTRGAAATAGAACAGCATAGAACAAAAATAGAGGAACTGAGACAACATCTGTCAAGGTGGGGGTTTACCACACCAGACAAAAAACATCAGAAAGAACCTCCATTCCTTTGGATGGGCTATGAACTCCATCCTGATAAATGGACAGTACAGCCTATAGTTCTGCCAGAAAAAGATAGCTGGACTGTCAATGACATACAGAAGTTAGTGGGGAAGTTGAATTGGGCAAGTCAGATTTAYGCAGGGATTAAAGTAAAGCAATTATGTAAACTCCTTAGGGGGACCAAGKCACTAACAGAAATAATACCACTAACAAGAGAAGCAGAGCTAGAACTGGCAGAAAACAGGGAAATTCTAAAAGAACCAGTACATGGAGTGTATTATGATCCAACAAAAGACTTAATAGCAGAAATACAGAAGCAGGGGCAAGGC000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 

  return new_RT_Seq_li

def load_HIV_csv_data(path):
  train_data_df=pd.read_csv(path,encoding='utf8')

  train_data_df=train_data_df.dropna()
  # print("train_data_df",train_data_df.shape)
  # (920, 6)

  train_data_wo_id_df=train_data_df.iloc[:,1:]
  # print("train_data_wo_id_df",train_data_wo_id_df.shape)
  # (920, 5)

  return train_data_wo_id_df

def count_num_seq_containing_B_or_H(train_data_wo_id_df):
  PR_seq_data_temp=train_data_wo_id_df.iloc[:,1]
  # print("PR_seq_data_temp",PR_seq_data_temp.shape)
  # (920,)

  # print("PR_seq_data_temp",PR_seq_data_temp)
  # afaf

  freq_of_char_B=0
  freq_of_char_H=0
  for one_PR in PR_seq_data_temp:
    # print("one_PR",one_PR)
    # CCTCAAATCACTCTTTGGCAACGACCCCTCGTCCCAATAAGGATAGGGGGGCAACTAAAGGAAGCYCTATTAGATACAGGAGCAGATGATACAGTATTAGAAGACATGGAGTTGCCAGGAAGATGGAAACCAAAAATGATAGGGGGAATTGGAGGTTTTATCAAAGTAARACAGTATGATCAGRTACCCATAGAAATCTATGGACATAAAGCTGTAGGTACAGTATTAATAGGACCTACACCTGTCAACATAATTGGAAGAAATCTGTTGACTCAGCTTGGTTGCACTTTAAATTTY

    if "H" in str(one_PR):
      freq_of_char_H=freq_of_char_H+1

    if "B" in str(one_PR):
      freq_of_char_B=freq_of_char_B+1
    
  # print("freq_of_char_B",freq_of_char_B)
  # print("freq_of_char_H",freq_of_char_H)
  # 1
  # 3

  return freq_of_char_B,freq_of_char_H

def get_indices_containing_B_or_H(train_data_wo_id_df):
  B_mask=train_data_wo_id_df.iloc[:,1].str.contains('B')
  # print("B_mask",B_mask)

  B_mask_idx=B_mask.index[B_mask==True].tolist()
  # print("B_mask_idx",B_mask_idx)
  # [25]

  H_mask=train_data_wo_id_df.iloc[:,1].str.contains('H')
  # print("H_mask",H_mask)

  H_mask_idx=H_mask.index[H_mask==True].tolist()
  # print("H_mask_idx",H_mask_idx)
  # [43, 199, 843]

  return B_mask_idx,H_mask_idx

def resolve_label_imbalance_on_resp(train_data_wo_id_df):
  resp_data=train_data_wo_id_df.iloc[:,0]
  trn_data_resp_0=train_data_wo_id_df[resp_data==0]
  trn_data_resp_1=train_data_wo_id_df[resp_data==1]
  # print("trn_data_resp_0",trn_data_resp_0.shape)
  # (733, 5)
  # print("trn_data_resp_1",trn_data_resp_1.shape)
  # (186, 5)
  
  trn_data_resp_1=resample(trn_data_resp_1,replace=True,n_samples=trn_data_resp_0.shape[0],random_state=123)
  # print("trn_data_resp_1",trn_data_resp_1.shape)
  # (733, 5)

  train_data_wo_id_df=pd.concat([trn_data_resp_0,trn_data_resp_1])
  # print("train_data_wo_id_df",train_data_wo_id_df.shape)
  # (1466, 5)

  # print("train_data_wo_id_df",train_data_wo_id_df.head(5))

  train_data_wo_id_df=shuffle(train_data_wo_id_df)
  # print("train_data_wo_id_df",train_data_wo_id_df.head(5))

  return train_data_wo_id_df



