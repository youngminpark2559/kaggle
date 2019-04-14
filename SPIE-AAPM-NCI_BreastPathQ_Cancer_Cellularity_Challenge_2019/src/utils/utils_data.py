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
from xml.dom import minidom
from bs4 import BeautifulSoup

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
    # ['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/18/47b5e2d534dde96fd2d2233f8e9bf5db64dcf480.tif\n',
    #  '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/52/d04fc062e2a62b0d8a6a2d4f7566eb319426c18d.tif\n',
    #  '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/07/1d322a3d9f821b372dc33287dc83ee250a985857.tif\n',
    #  '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/13/370c9f1c6bd16da927ed53da292568f4320dc105.tif\n',
    #  '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/38/960d671408870f513c30565db2c62f60ccdbe635.tif\n',
    #  '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/18/492a39b2940176a63f7c55b21ec49044a4068a2f.tif\n',
    #  '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/44/ada807e5f7789633847e003a414c50959352284b.tif\n',
    #  '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/42/a67ce46ef58ed08bcb9222809d7fc7df73cd2e59.tif\n',
    #  '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/01/05ba85446a11b0e09c23de3a093ca586eaf89b83.tif\n',
    #  '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/21/5434adc3a8bd96ecd35d73322f682aab253eea48.tif\n']

    labels_of_imgs=dataset_bs_paths[1]
    # print("labels_of_imgs",labels_of_imgs)
    # [0 1 1 0 0 1 0 1 1 0]

    # ================================================================================
    # Load images
    dataset_bs_paths=[]
    for x in paths_of_imgs:
      loaded_img=Image.open(x.replace("\n",""))
      # When you don't use gt image
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

    # region augmentation on group
    # aug_pipeline=Augmentor.DataPipeline(dataset_bs_paths,labels_of_imgs)

    # # ================================================================================
    # # crop_by_size
    # aug_pipeline.crop_by_size(probability=1.0,width=48,height=48,centre=True)

    # # ================================================================================
    # # rotate
    # aug_pipeline.rotate(probability=0.5,max_left_rotation=6,max_right_rotation=7)

    # # ================================================================================
    # # flip_random
    # aug_pipeline.flip_random(probability=0.5)

    # # ================================================================================
    # # Random sample images
    # sampled_trn_and_rgt_imgs_li,label_values=aug_pipeline.sample(int(args.batch_size))
    # print("sampled_trn_and_rgt_imgs_li",len(sampled_trn_and_rgt_imgs_li))
    # # sampled_trn_and_rgt_imgs_li 5

    # print("label_values",label_values)
    # # label_values [0, 0, 0, 0, 1]

    # # --------------------------------------------------------------------------------
    # sampled_trn_and_rgt_imgs=np.array(sampled_trn_and_rgt_imgs_li)/255.0
    # # print("sampled_trn_and_rgt_imgs",sampled_trn_and_rgt_imgs.shape)
    # # sampled_trn_and_rgt_imgs (5, 1, 64, 64, 3)

    # sampled_trn_imgs=sampled_trn_and_rgt_imgs[:,0,:,:,:]

    # sampled_trn_imgs_tc=sampled_trn_imgs.transpose(0,3,1,2)
    # # print("sampled_trn_imgs_tc",sampled_trn_imgs_tc.shape)
    # # (11, 3, 96, 96)
    # endregion 
    
    # ================================================================================
    # c kind_of_DA: you create list which contains kind of data augmentation
    # kind_of_DA=["no_DA","ud","lr","p3","p6","p9","n3","n6","n9"]
    kind_of_DA=["no_DA","ud","lr","p3","p6","n3","n6"]

    # c chosen_DA: you get chosen kind of data augmentation
    chosen_DA=np.random.choice(kind_of_DA,1,replace=False)[0]
    # print("chosen_DA",chosen_DA)
    # lr

    dataset_bs_img_np=np.array(dataset_bs_paths).squeeze()
    # print("dataset_bs_img_np",dataset_bs_img_np.shape)
    # (10, 96, 96, 3)

    # ================================================================================
    # @ Flip or rotate

    after_aug_imgs=[]
    for one_idx in range(dataset_bs_img_np.shape[0]):
      one_img=dataset_bs_img_np[one_idx,:,:,:]
      # print("one_img",one_img.shape)
      # (48, 48, 3)
      # scipy.misc.imsave('./tumor_before_DA_'+str(one_idx)+'.png',one_img)

      if chosen_DA=="ud":
        one_img=np.flipud(one_img)
      elif chosen_DA=="lr":
        one_img=np.fliplr(one_img)
      elif chosen_DA=="p3":
        one_img=scipy.ndimage.interpolation.rotate(one_img,angle=3,reshape=True,mode="reflect")
      elif chosen_DA=="p6":
        one_img=scipy.ndimage.interpolation.rotate(one_img,angle=6,reshape=True,mode="reflect")
      elif chosen_DA=="p9":
        one_img=scipy.ndimage.interpolation.rotate(one_img,angle=9,reshape=True,mode="reflect")
      elif chosen_DA=="n3":
        one_img=scipy.ndimage.interpolation.rotate(one_img,angle=-3,reshape=True,mode="reflect")
      elif chosen_DA=="n6":
        one_img=scipy.ndimage.interpolation.rotate(one_img,angle=-6,reshape=True,mode="reflect")
      elif chosen_DA=="n9":
        one_img=scipy.ndimage.interpolation.rotate(one_img,angle=-9,reshape=True,mode="reflect")
      else:
          pass
      
      # ================================================================================
      one_img=np.clip(one_img/255.0,0.,1.)

      # ================================================================================
      # @ Resize image to (224,224,3)

      one_img=resize(one_img,(224,224))
      # print("one_img",one_img.shape)
      # (224, 224, 3)

      # ================================================================================
      # scipy.misc.imsave('./tumor_after_DA_'+str(chosen_DA)+str(one_idx)+'.png',one_img)

      after_aug_imgs.append(one_img)

    after_aug_imgs_np=np.array(after_aug_imgs)

    # ================================================================================
    # @ Center crop

    # after_aug_imgs_np=after_aug_imgs_np[:,24:72,24:72,:]
    # print("after_aug_imgs_np",after_aug_imgs_np.shape)
    # (10, 48, 48, 3)

    # ================================================================================
    after_aug_imgs_np=after_aug_imgs_np.transpose(0,3,1,2)
    # print("after_aug_imgs_np",after_aug_imgs_np.shape)
    # (40, 3, 224, 224)

    # ================================================================================
    # print("labels_of_imgs",labels_of_imgs)
    # [1 1 1 0 0 0 0 0 0 0]

    labels_of_imgs_np=np.array(labels_of_imgs).astype("float32")
    # print("labels_of_imgs_np",labels_of_imgs_np)

    return after_aug_imgs_np,labels_of_imgs_np

  except:
    print(traceback.format_exc())
    print("Error when loading images")
    print("path_to_be_loaded",path_to_be_loaded)

def load_xml_file(xml_file_path):

  fp=open(xml_file_path,"r")
  soup=BeautifulSoup(fp,"html.parser")
  # print("soup",soup)

  whole_xml_data=[]
  # findAll로 해당되는 TAG를 검색
  for graphic_tag_one in soup.findAll('graphic'):

    # ================================================================================
    gra_type=graphic_tag_one["type"]
    gra_name=graphic_tag_one["name"]
    # print("gra_type",gra_type)
    # print("gra_name",gra_name)
    # point
    # Region 2

    # ================================================================================
    pen_color=graphic_tag_one.pen["color"]
    # print('pen_color',pen_color)
    # #005500ff

    # ================================================================================
    point_pairs=getattr(graphic_tag_one,'point-list')
    # print("point_pairs",point_pairs)
    # <point-list>
    # <point>32,18</point>
    # <point>51,13</point>

    # ================================================================================
    one_data=[gra_type,gra_name,pen_color,point_pairs]

    # ================================================================================
    whole_xml_data.append(one_data)

  # ================================================================================
  # print("whole_xml_data",whole_xml_data)
  # [['point', 'Region 2', <point-list> <point>32,18</point>
  
  return whole_xml_data















