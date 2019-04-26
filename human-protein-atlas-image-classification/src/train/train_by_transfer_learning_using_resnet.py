import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys,os,copy,argparse
import time,timeit,datetime
import glob,natsort
import cv2
from PIL import Image
from skimage.transform import resize
from skimage.restoration import (denoise_tv_chambolle,denoise_bilateral,
                                 denoise_wavelet,estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,\
                            precision_score,recall_score,fbeta_score,f1_score,roc_curve
import scipy.misc
import scipy.special
from skimage.viewer import ImageViewer
from random import shuffle
import gc
import Augmentor
import traceback
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
from torch.autograd import gradcheck
from torch.optim.lr_scheduler import StepLR

# ================================================================================
from src.utils import utils_common as utils_common
from src.utils import utils_image as utils_image
from src.utils import utils_net as utils_net
from src.utils import utils_pytorch as utils_pytorch
from src.utils import utils_data as utils_data
from src.utils import utils_hook_functions as utils_hook_functions
from src.utils import utils_visualize_gradients as utils_visualize_gradients

from src.utils_for_dataset import custom_ds as custom_ds
from src.utils_for_dataset import custom_ds_test as custom_ds_test

from src.loss_functions import loss_functions_module as loss_functions_module

from src.api_model import model_api_module as model_api_module
from src.api_text_file_path import text_file_path_api_module as text_file_path_api_module

from src.utils_analyzing_result import grad_cam as grad_cam

# ================================================================================
def train(args):

  epoch=int(args.epoch)
  batch_size=int(args.batch_size)
  # print("epoch",epoch)
  # print("batch_size",batch_size)
  # 9
  # 2

  # ================================================================================
  text_file_instance=text_file_path_api_module.Path_Of_Text_Files(args)

  txt_of_image_data=tumor_trn=text_file_instance.image_data
  txt_of_label_data=text_file_instance.label_data
  # print("txt_of_image_data",txt_of_image_data)
  # print("txt_of_label_data",txt_of_label_data)
  # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/Path_of_train_images.txt
  # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train.csv

  # ================================================================================
  # c loss_list: list which will stores loss values to plot loss
  loss_list=[]

  # ================================================================================
  # c model_api_instance: instance of model API
  model_api_instance=model_api_module.Model_API_class(args)

  # ================================================================================
  # # @ Test Grad CAM
  # imgs=["/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/33/82d4d190d2fed1be255fc3bac36a37c860bb31c0.tif",
  #       "/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/33/82a5300cd61628fb9bae332cdb7d5e7e37b1fb36.tif"]
  # grad_cam.initialize_grad_cam(model=model_api_instance.gen_net,list_of_img_paths=imgs,args=args)

  # ================================================================================
  if args.task_mode=="train": # If you're in train mode
    
    # ================================================================================
    # @ Configure learning rate scheduler

    # Update learning rate 4 times during entire epochs
    # For example, if you use 10 epochs, int(10/4), 1 2 / 3 4 / 5 6 / 7 8 / 9 10
    # 0-1 epochs: 0.001 -> 2-3 epochs: 0.0001 -> 4-5 epochs: 0.00001 -> 5-6 epochs: 0.000001

    scheduler=StepLR(model_api_instance.optimizer,step_size=int(epoch/4),gamma=0.1)

    # ================================================================================
    for one_ep in range(epoch): # @ Iterates all epochs

      # c dataset_inst_trn: dataset instance of tumor
      dataset_inst_trn=custom_ds.Custom_DS(
        txt_containing_paths=txt_of_image_data,txt_containing_labels=txt_of_label_data,is_train=True,args=args)
      
      # Test iterator
      # iter_dataset_inst_trn=iter(dataset_inst_trn)
      # trn=next(iter_dataset_inst_trn)
      # print("trn",trn)

      # ================================================================================
      # c dataloader_trn: create dataloader
      dataloader_trn=torch.utils.data.DataLoader(
        dataset=dataset_inst_trn,batch_size=batch_size,shuffle=False,num_workers=3)
      
      # # c dataloader_trn_iter: iterator of dataloader
      # dataloader_trn_iter=iter(dataloader_trn)
      # # Test dataloader
      # pairs=next(dataloader_trn_iter)
      # # print("pairs",pairs)

      # ================================================================================
      # c num_imgs_trn: number of train image
      num_imgs_trn=len(dataset_inst_trn)
      # print("num_imgs_trn",num_imgs_trn)
      # 27964

      args.__setattr__("num_imgs_trn",num_imgs_trn)
      # print("args",args)
      
      # ================================================================================
      # print("Current batch size:",batch_size)
      # print("Possible batch size:",list(utils_common.divisorGenerator(num_imgs_trn)))
      # assert str(num_imgs_trn/batch_size).split(".")[-1]==str(0),"Check batch size, currently it's incorrect"

      # ================================================================================
      # @ If you don't use Augmentor
      if args.use_augmentor=="False":
        pass      

      else: # @ If you use Augmentor

        # @ Iterate all images in dataset during single epoch
        for idx,data in enumerate(dataloader_trn):
          # print("data",data)
          # [[('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/5f3ad194-bbb4-11e8-b2ba-ac1f6b6435d0_blue.png\n',
          #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/3a846fd6-bba0-11e8-b2b9-ac1f6b6435d0_blue.png\n'),
          #   ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/5f3ad194-bbb4-11e8-b2ba-ac1f6b6435d0_green.png\n',
          #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/3a846fd6-bba0-11e8-b2b9-ac1f6b6435d0_green.png\n'),
          #   ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/5f3ad194-bbb4-11e8-b2ba-ac1f6b6435d0_red.png\n',
          #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/3a846fd6-bba0-11e8-b2b9-ac1f6b6435d0_red.png\n'),
          #   ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/5f3ad194-bbb4-11e8-b2ba-ac1f6b6435d0_yellow.png\n',
          #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/3a846fd6-bba0-11e8-b2b9-ac1f6b6435d0_yellow.png\n')],
          #  [('5f3ad194-bbb4-11e8-b2ba-ac1f6b6435d0','3a846fd6-bba0-11e8-b2b9-ac1f6b6435d0'),
          #   ('12 0','25 11 2')]]
          
          # c paths_of_imgs: paths of images
          paths_of_imgs=data[0]
          # print("paths_of_imgs",paths_of_imgs)
          # [('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/c3a572e8-bbbd-11e8-b2ba-ac1f6b6435d0_blue.png\n',
          #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00456fd2-bb9b-11e8-b2b9-ac1f6b6435d0_blue.png\n'),
          #  ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/c3a572e8-bbbd-11e8-b2ba-ac1f6b6435d0_green.png\n',
          #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00456fd2-bb9b-11e8-b2b9-ac1f6b6435d0_green.png\n'),
          #  ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/c3a572e8-bbbd-11e8-b2ba-ac1f6b6435d0_red.png\n',
          #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00456fd2-bb9b-11e8-b2b9-ac1f6b6435d0_red.png\n'),
          #  ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/c3a572e8-bbbd-11e8-b2ba-ac1f6b6435d0_yellow.png\n',
          #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00456fd2-bb9b-11e8-b2b9-ac1f6b6435d0_yellow.png\n')]

          # images=[[im_1, mask_1_a, mask_1_b],
          #         [im_2, mask_2_a, mask_2_b],
          #         ...,
          #         [im_n, mask_n_a, mask_n_b]]

          # y is label like [0,1,1,0,1,1] when you use 6 images which have binary class

          B_imgs=list(paths_of_imgs[0])
          G_imgs=list(paths_of_imgs[1])
          R_imgs=list(paths_of_imgs[2])
          Y_imgs=list(paths_of_imgs[3])
          
          zipped_paths=list(zip(B_imgs,G_imgs,R_imgs,Y_imgs))
          # print("zipped_paths",zipped_paths)
          # [('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/06a9f3e8-bba5-11e8-b2ba-ac1f6b6435d0_blue.png\n',
          #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/06a9f3e8-bba5-11e8-b2ba-ac1f6b6435d0_green.png\n',
          #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/06a9f3e8-bba5-11e8-b2ba-ac1f6b6435d0_red.png\n',
          #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/06a9f3e8-bba5-11e8-b2ba-ac1f6b6435d0_yellow.png\n'),
          #  ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/bc7aabca-bbc0-11e8-b2bb-ac1f6b6435d0_blue.png\n',
          #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/bc7aabca-bbc0-11e8-b2bb-ac1f6b6435d0_green.png\n',
          #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/bc7aabca-bbc0-11e8-b2bb-ac1f6b6435d0_red.png\n',
          #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/bc7aabca-bbc0-11e8-b2bb-ac1f6b6435d0_yellow.png\n')]

          # ================================================================================
          # c labels_in_scalar: label values of images
          labels_in_scalar=np.array(data[1][1])
          # print("labels_in_scalar",labels_in_scalar)
          # ['19 0' '25 6 0']

          # ================================================================================
          # c bs_pa_tumor_d: batchsized paths of tumor dataset
          bs_pa_tumor_d=[zipped_paths,labels_in_scalar]
          # print("bs_pa_tumor_d",bs_pa_tumor_d)
          # [[('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/ab03b420-bbbd-11e8-b2ba-ac1f6b6435d0_blue.png\n',
          #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/ab03b420-bbbd-11e8-b2ba-ac1f6b6435d0_green.png\n',
          #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/ab03b420-bbbd-11e8-b2ba-ac1f6b6435d0_red.png\n',
          #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/ab03b420-bbbd-11e8-b2ba-ac1f6b6435d0_yellow.png\n'),
          #   ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/815d3f74-bbc2-11e8-b2bb-ac1f6b6435d0_blue.png\n',
          #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/815d3f74-bbc2-11e8-b2bb-ac1f6b6435d0_green.png\n',
          #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/815d3f74-bbc2-11e8-b2bb-ac1f6b6435d0_red.png\n',
          #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/815d3f74-bbc2-11e8-b2bb-ac1f6b6435d0_yellow.png\n')],
          #  array(['22','0 21'],dtype='<U4')]


          # ================================================================================
          # @ Perform data augmentation

          sampled_trn_imgs,label_values=utils_data.use_augmetor_for_tumor_data(bs_pa_tumor_d,args)
          
          # print("sampled_trn_imgs",sampled_trn_imgs.shape)
          # (2, 4, 224, 224)
          
          # print("label_values",label_values)
          # [[4], [14]]

          # print("label_values",np.array(label_values).shape)
          # (2, 2)

          # ================================================================================
          oh_label_arr=utils_common.one_hot_label(batch_size,label_values)
          # print("oh_label_arr",oh_label_arr)
          # [[0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
          #  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

          # ================================================================================
          trn_imgs_tcv=utils_pytorch.get_Variable(sampled_trn_imgs)
          # print("trn_imgs_tcv",trn_imgs_tcv.shape)
          # torch.Size([2, 4, 224, 224])

          # ================================================================================
          # @ Remove existing gradients
          model_api_instance.remove_existing_gradients_before_starting_new_training()
          
          # ================================================================================
          # @ c predicted_labels: pass input images and get predictions
          predicted_labels=model_api_instance.gen_net(trn_imgs_tcv)
          # print("predicted_labels",predicted_labels)
          # tensor([[-0.2858, -0.7700, -0.0600,  0.3553,  0.0367, -0.4130,  0.3102, -0.2443,
          #          -0.1775, -0.1839,  0.0499, -0.1489, -0.9805,  0.1817, -0.0504,  0.8930,
          #          -0.4017, -0.1899,  0.0937, -0.3465,  0.2830, -0.2755,  0.4233, -0.1301,
          #           1.1688,  0.2110,  0.1423, -0.3933],
          #         [-0.2858, -0.7700, -0.0600,  0.3553,  0.0367, -0.4130,  0.3102, -0.2443,
          #          -0.1775, -0.1839,  0.0499, -0.1489, -0.9805,  0.1817, -0.0504,  0.8930,
          #          -0.4017, -0.1899,  0.0937, -0.3465,  0.2830, -0.2755,  0.4233, -0.1301,
          #           1.1688,  0.2110,  0.1423, -0.3933]], device='cuda:0',grad_fn=<AddmmBackward>)

          label_tc=Variable(torch.tensor(oh_label_arr,device=predicted_labels.device).float())
          # print("label_tc",label_tc)
          # tensor([[0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          #         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], 
          #         device='cuda:0',dtype=torch.float16)

          # ================================================================================
          # @ Calculate loss values

          loss_val=loss_functions_module.FocalLoss(predicted_labels,label_tc)
          print("loss_val",loss_val)
          afaf
          # tensor(6.5246, device='cuda:0', grad_fn=<MeanBackward1>)
                    
          # ================================================================================
          # @ Calculate gradient values through backpropagation
          loss_val.backward()

          # ================================================================================
          # @ Update parameters of the network based on gradients
          model_api_instance.optimizer.step()

          # ================================================================================
          # @ If you want to print loss
          if args.use_loss_display=="True":
            if idx%int(args.leapping_term_when_displaying_loss)==0:
              print("Epoch:",one_ep,", Batch:",idx)
              print("loss_from_one_batch",loss_val.item())
          
          loss_list.append(loss_val.item())

          # ================================================================================
          # @ Save model after every batch you configure 
          # by using args.leapping_term_when_saving_model_after_batch
          if idx%int(args.leapping_term_when_saving_model_after_batch)==0:
            num_batch="batch_"+str(idx)
            model_api_instance.save_model_after_epoch(num_batch)

      # ================================================================================
      # @ Save model after epoch
      num_epoch="epoch_"+str(one_ep)
      model_api_instance.save_model_after_epoch(num_epoch)

      # ================================================================================
      # @ Update learning rate

      scheduler.step()
      # print("scheduler.base_lrs",scheduler.base_lrs)

    # ================================================================================
    # @ Plot loss value
    plt.plot(loss_list)
    plt.savefig("loss.png")
    plt.show()
  
  # ================================================================================
  elif args.task_mode=="validation":
    with torch.no_grad(): # @ Use network without calculating gradients
      # tumor_trn=args.dir_where_text_file_for_image_paths_is_in+"/tumor_trn.txt"
      # tumor_lbl=args.dir_where_text_file_for_image_paths_is_in+"/train_labels.csv"
      # print("tumor_trn",tumor_trn)
      # print("tumor_lbl",tumor_lbl)
      # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/tumor_trn.txt
      # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_labels.csv

      # ================================================================================
      # @ Dataset and Dataloader

      # @ c dataset_inst_test_tumor: dataset instance of tumor dataset
      dataset_inst_test_tumor=custom_ds.custom_ds(
        txt_containing_paths=tumor_trn,txt_containing_labels=tumor_lbl,is_train=False,args=args)

      # @ c dataloader_tumor_test: dataloader instance of tumor dataset
      dataloader_tumor_test=torch.utils.data.DataLoader(
        dataset=dataset_inst_test_tumor,batch_size=batch_size,shuffle=False,num_workers=3)
      
      # ================================================================================
      # @ c num_imgs_test: number of entire test images
      num_imgs_test=len(dataset_inst_test_tumor)

      # ================================================================================
      # @ Create network and optimizer
      if args.train_method=="train_by_transfer_learning_using_resnet":
        model_api_instance=model_api_module.Model_API_class(args)
      
      # ================================================================================
      predicted_values=[]
      true_values=[]
      img_paths=[]

      # ================================================================================
      # @ Iterate all batches (batch1+batch2+...+batchn=entire images)
      for idx,data in enumerate(dataloader_tumor_test):
        # print("idx",idx)
        # print("data",data)
        # [('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/e693f9ac4097289c317831960514b78701999cd9.tif\n',
        #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/e6941f6c6825e7c409b9364e2fb6c2d629df8a76.tif\n',),
        #  [('e693f9ac4097289c317831960514b78701999cd9','e6941f6c6825e7c409b9364e2fb6c2d629df8a76'),tensor([1,0])]]
        
        # ================================================================================
        # @ c imgs: paths of validation images
        imgs=data[0]

        img_paths.extend(imgs)

        # @ c imgs: labels to validation images
        lbls=data[1][1].numpy()

        # @ c num_imgs: number of validation image in one batch
        num_imgs=lbls.shape[0]
        # print("num_imgs",num_imgs)
        # 11
        # @ Load images from paths

        # ================================================================================
        test_imgs_list=[]
        for one_img_path in imgs:
          one_loaded_img=utils_image.load_img(one_img_path)
          # print("one_loaded_img",one_loaded_img.shape)
          # (96, 96, 3)

          one_loaded_img=resize(one_loaded_img,(224,224))

          test_imgs_list.append(one_loaded_img)

        # ================================================================================
        test_imgs_np=np.array(test_imgs_list).transpose(0,3,1,2)
        
        # @ If you want to use center (48,48) image from (96,96) image
        # test_imgs_np=test_imgs_np[:,:,24:72,24:72]
        # print("test_imgs_np",test_imgs_np.shape)
        # (11, 3, 48, 48)

        test_imgs_tc=Variable(torch.Tensor(test_imgs_np).cuda())

        # ================================================================================
        # @ Make predictions

        prediction=model_api_instance.gen_net(test_imgs_tc)
        # print("prediction",prediction)
        # tensor([[-2.0675],
        #         [-2.9296],

        sigmoid=torch.nn.Sigmoid()

        prediction_np=sigmoid(prediction).cpu().numpy()

        # ================================================================================
        # @ Make predicted labels

        prediction_np=np.where(prediction_np>0.5,1,0).squeeze()
        # print("prediction_np",prediction_np)
        # [0 0 1 0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 1 0 1 1 0 0]
        # print("lbls",lbls)
        # [1 0 0 0 0 1 0 0 1 0 0 1 1 0 1 0 1 0 0 0 0 0 0 0 1 0 1 1 1 0]

        predicted_values.extend(prediction_np)
        
        true_values.extend(lbls)
      
      # ================================================================================
      y_true=true_values
      y_pred=predicted_values

      # ================================================================================
      # @ Binary Confusion Matrix

      b_c_mat=confusion_matrix(true_values,predicted_values,labels=[0,1])
      # print("b_c_mat",b_c_mat)
      # [[30  2]
      #  [ 0 68]]

      # True Positive (Tumor pic is predicted as tumor)      False Negative (Tumor pic is predicted as non-tumor)
      # False Positive (Non-tumor pic is predicted as tumor) True Negative (Non-tumor pic is predicted as non-tumor)
      
      # ================================================================================
      # @ metric report
      
      report=classification_report(y_true,y_pred,target_names=['class Non tumor (neg)', 'class Tumor (pos)'])
      # print(report)
      #                        precision    recall  f1-score   support

      # class Non tumor (neg)       0.97      1.00      0.99        68
      #     class Tumor (pos)       1.00      0.94      0.97        32

      #             micro avg       0.98      0.98      0.98       100
      #             macro avg       0.99      0.97      0.98       100
      #          weighted avg       0.98      0.98      0.98       100

      # ================================================================================
      print("accuracy_score",accuracy_score(y_true,y_pred))
      # 0.98

      print("precision_score",precision_score(y_true,y_pred))
      # 1.0

      print("recall_score",recall_score(y_true,y_pred))
      # 0.9375

      # print("fbeta_score",fbeta_score(y_true, y_pred, beta))
      
      print("f1_score",fbeta_score(y_true,y_pred,beta=1))
      # 0.967741935483871

      # ================================================================================
      # @ ROC curve
      fpr,tpr,thresholds=roc_curve(y_true,y_pred)
      plt.plot(fpr,tpr,'o-',label="Logistic Regression")
      plt.title('Receiver operating characteristic example')
      plt.show()

  elif args.task_mode=="submission":
    with torch.no_grad(): # @ Use network without calculating gradients
      
      sub_ds=custom_ds_test.custom_ds_Submission()
      print("sub_ds",sub_ds)

      sub_dl=torch.utils.data.DataLoader(
        dataset=sub_ds,batch_size=batch_size,shuffle=False,num_workers=3)
      print("sub_dl",sub_dl)

      # ================================================================================
      # @ c num_imgs_test: number of entire test images

      num_imgs_test=len(sub_ds)

      # ================================================================================
      # @ Create network and optimizer

      if args.train_method=="train_by_transfer_learning_using_resnet":
        model_api_instance=model_api_module.Model_API_class(args)

      # ================================================================================
      label_submission=pd.read_csv("/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/sample_submission.csv",encoding='utf8')
      base_names=label_submission.iloc[:,0].tolist()
      # print("base_names",base_names)

      # ================================================================================
      predicted_values=[]
      # @ Iterate all batches (batch1+batch2+...+batchn=entire images)
      for idx,data in enumerate(sub_dl):
        # print("idx",idx)
        # print("data",data)
        # 0
        # ['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/test/0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5.tif',
        
        imgs=data

        # ================================================================================
        test_imgs_list=[]
        for one_img_path in imgs:
          one_loaded_img=utils_image.load_img(one_img_path)
          # print("one_loaded_img",one_loaded_img.shape)
          # (96, 96, 3)

          one_loaded_img=resize(one_loaded_img,(224,224))

          test_imgs_list.append(one_loaded_img)

        # ================================================================================
        test_imgs_np=np.array(test_imgs_list).transpose(0,3,1,2)
        
        # @ If you want to use center (48,48) image from (96,96) image
        # test_imgs_np=test_imgs_np[:,:,24:72,24:72]
        # print("test_imgs_np",test_imgs_np.shape)
        # (11, 3, 48, 48)

        test_imgs_tc=Variable(torch.Tensor(test_imgs_np).cuda())
        # print("test_imgs_tc",test_imgs_tc.shape)
        # torch.Size([30, 3, 224, 224])

        # ================================================================================
        # @ Make predictions
        prediction=model_api_instance.gen_net(test_imgs_tc)
        # print("prediction",prediction)
        # tensor([[-2.0675],
        # ...
        #         [-1.2222]], device='cuda:0')

        sigmoid=torch.nn.Sigmoid()

        prediction_np=sigmoid(prediction).cpu().numpy()

        # ================================================================================
        # @ Make predicted labels

        prediction_np=np.where(prediction_np>0.5,1,0).squeeze()
        # print("prediction_np",prediction_np)
        # [0 0 1 0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 1 0 1 1 0 0]
        # print("lbls",lbls)
        # [1 0 0 0 0 1 0 0 1 0 0 1 1 0 1 0 1 0 0 0 0 0 0 0 1 0 1 1 1 0]

        predicted_values.extend(prediction_np)
     
      my_submission=pd.DataFrame({'id': base_names,'label': predicted_values})
      my_submission.to_csv('youngminpar2559_submission.csv',index=False)
