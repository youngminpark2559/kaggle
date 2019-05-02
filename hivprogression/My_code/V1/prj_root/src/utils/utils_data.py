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

# ================================================================================
from src.utils import utils_common as utils_common
from src.utils import utils_image as utils_image

# ================================================================================
def use_augmetor_for_data(dataset_bs_paths,args):
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
    # [('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/d84b9aa8-bbb6-11e8-b2ba-ac1f6b6435d0_blue.png',
    #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/d84b9aa8-bbb6-11e8-b2ba-ac1f6b6435d0_green.png',
    #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/d84b9aa8-bbb6-11e8-b2ba-ac1f6b6435d0_red.png',
    #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/d84b9aa8-bbb6-11e8-b2ba-ac1f6b6435d0_yellow.png'),
    #  ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/6d3a7a02-bb9f-11e8-b2b9-ac1f6b6435d0_blue.png',
    #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/6d3a7a02-bb9f-11e8-b2b9-ac1f6b6435d0_green.png',
    #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/6d3a7a02-bb9f-11e8-b2b9-ac1f6b6435d0_red.png',
    #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/6d3a7a02-bb9f-11e8-b2b9-ac1f6b6435d0_yellow.png')]

    # ================================================================================
    labels_of_imgs=dataset_bs_paths[1]
    # print("labels_of_imgs",labels_of_imgs)
    # ['0' '5 0']

    # ================================================================================
    labels_of_imgs_li=[]
    for one_lbl_set in labels_of_imgs:
      one_list=one_lbl_set.split(" ")
      # print("one_list",one_list)
      # ['0']

      one_list=list(map(int,one_list))
      # print("one_list",one_list)
      # [0]

      labels_of_imgs_li.append(one_list)

    # print("labels_of_imgs_li",labels_of_imgs_li)
    # [[0], [5, 0]]

    # ================================================================================
    # @ Load images

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

# ================================================================================
def get_k_folds(txt_of_image_data,txt_of_label_data):
  path_of_imgs,num_loaded_imgs=utils_common.return_path_list_from_txt(txt_of_image_data)

  path_of_imgs=[one_path.replace("\n","") for one_path in path_of_imgs]

  path_of_imgs_chunked=utils_common.chunk_proteins_by_4C(path_of_imgs)
  # print("path_of_imgs_chunked",path_of_imgs_chunked)
  # [['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_blue.png',
  #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_green.png',
  #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_red.png',
  #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_yellow.png'],


  train_index_set,validation_index_set=utils_common.split_by_k_folds(path_of_imgs_chunked)
  # print("train_index_set",train_index_set)
  # [array([    1,     2,     3, ..., 31063, 31064, 31065]), array([    0,     4,     6, ..., 31069, 31070, 31071]), array([    0,     1,     2, ..., 31069, 31070, 31071])]
  # print("validation_index_set",validation_index_set)
  # [array([    0,     4,     6, ..., 31069, 31070, 31071]), array([    1,     2,     3, ..., 31060, 31061, 31065]), array([   10,    11,    13, ..., 31062, 31063, 31064])]

  # ================================================================================
  train_path_k0=np.array(path_of_imgs_chunked)[train_index_set[0]]
  train_path_k1=np.array(path_of_imgs_chunked)[train_index_set[1]]
  train_path_k2=np.array(path_of_imgs_chunked)[train_index_set[2]]
  # print("train_path_k2",train_path_k2)
  # [['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_blue.png\n'
  #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_green.png\n'
  #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_red.png\n'
  #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_yellow.png\n']

  vali_path_k0=np.array(path_of_imgs_chunked)[validation_index_set[0]]
  vali_path_k1=np.array(path_of_imgs_chunked)[validation_index_set[1]]
  vali_path_k2=np.array(path_of_imgs_chunked)[validation_index_set[2]]
  # print("vali_path_k2",vali_path_k2)
  # [['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/002ff91e-bbb8-11e8-b2ba-ac1f6b6435d0_blue.png\n'
  #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/002ff91e-bbb8-11e8-b2ba-ac1f6b6435d0_green.png\n'
  #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/002ff91e-bbb8-11e8-b2ba-ac1f6b6435d0_red.png\n'
  #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/002ff91e-bbb8-11e8-b2ba-ac1f6b6435d0_yellow.png\n']

  # ================================================================================
  loaded_label_data=pd.read_csv(txt_of_label_data,encoding='utf8')

  loaded_label_data_sorted=loaded_label_data.sort_values(by=["Id"],ascending=True)
  # print("loaded_label_data_sorted",loaded_label_data_sorted.head())
  #                                      Id   Target
  # 0  00070df0-bbc3-11e8-b2bc-ac1f6b6435d0     16 0
  # 1  000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0  7 1 2 0
  # print("loaded_label_data_sorted",loaded_label_data_sorted.shape)
  # (31072, 2)

  # c loaded_label_data_sorted_list: label into list
  loaded_label_data_sorted_list=loaded_label_data_sorted.iloc[:,:].values.tolist()

  # print("loaded_label_data_sorted_list",loaded_label_data_sorted_list)
  # [['00070df0-bbc3-11e8-b2bc-ac1f6b6435d0', '16 0'], 
  #  ['000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0', '7 1 2 0'], 
  #  ['000a9596-bbc4-11e8-b2bc-ac1f6b6435d0', '5'], 
  #  ['000c99ba-bba4-11e8-b2b9-ac1f6b6435d0', '1'],

  loaded_label_data_sorted_np=np.array(loaded_label_data_sorted_list)
  
  train_label_k0=loaded_label_data_sorted_np[train_index_set[0]]
  train_label_k1=loaded_label_data_sorted_np[train_index_set[1]]
  train_label_k2=loaded_label_data_sorted_np[train_index_set[2]]
  # print("train_label_k2",train_label_k2)
  # [['00070df0-bbc3-11e8-b2bc-ac1f6b6435d0' '16 0']
  #  ['000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0' '7 1 2 0']

  vali_label_k0=loaded_label_data_sorted_np[validation_index_set[0]]
  vali_label_k1=loaded_label_data_sorted_np[validation_index_set[1]]
  vali_label_k2=loaded_label_data_sorted_np[validation_index_set[2]]
  # print("vali_label_k2",vali_label_k2)
  # [['002ff91e-bbb8-11e8-b2ba-ac1f6b6435d0' '23']
  #  ['00301238-bbb2-11e8-b2ba-ac1f6b6435d0' '21']

  # ================================================================================
  train_k=[train_path_k0,train_path_k1,train_path_k2]
  vali_k=[vali_path_k0,vali_path_k1,vali_path_k2]
  train_lbl_k=[train_label_k0,train_label_k1,train_label_k2]
  vali_lbl_k=[vali_label_k0,vali_label_k1,vali_label_k2]

  # ================================================================================
  return train_k,vali_k,train_lbl_k,vali_lbl_k

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



