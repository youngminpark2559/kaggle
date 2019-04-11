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

from src.utils_for_dataset import dataset_tumor as dataset_tumor
from src.utils_for_dataset import dataset_tumor_test as dataset_tumor_test

from src.model_api import model_api_module as model_api_module

from src.text_file_path_api import text_file_path_api_module as text_file_path_api_module

from src.utils_analyzing_result import grad_cam as grad_cam

# ================================================================================
def train(args):

  epoch=int(args.epoch)
  batch_size=int(args.batch_size)
  # print("epoch",epoch)
  # print("batch_size",batch_size)
  # 200
  # 22

  # ================================================================================
  text_file_instance=text_file_path_api_module.Path_Of_Text_Files(args)
  tumor_trn=text_file_instance.tumor_trn
  tumor_lbl=text_file_instance.tumor_lbl
  # print("tumor_trn",tumor_trn)
  # print("tumor_lbl",tumor_lbl)
  # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/trn_txt_file_processed.txt
  # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_labels.csv

  # ================================================================================
  # c tumor_loss_temp: list which will stores loss values to plot loss
  tumor_loss_temp=[]

  # ================================================================================
  # c model_api_instance: instance of model API
  model_api_instance=model_api_module.Model_API_class(args)

  # # ================================================================================
  # # @ Test Grad CAM
  # imgs=["/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/my_model/my_model_new_architecture/prj_root/src/utils_analyzing_result/Grad_CAM_output/02dcc96b7dfd481e806d86d3344a433340f9ec21.tif",
  #       "/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/test/ffcc29cf0e363737b577d1db470df0bb1adf7957.tif"]
  # grad_cam.initialize_grad_cam(model=model_api_instance.gen_net,list_of_img_paths=imgs,args=args)
  # # afaf 1: grad_cam.initialize_grad_cam(model=model_api_instance.gen_net,list_of_img_paths=imgs,args=args)

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

      # c dataset_inst_trn_tumor: dataset instance of tumor
      dataset_inst_trn_tumor=dataset_tumor.Dataset_Tumor(
        txt_containing_paths=tumor_trn,txt_containing_labels=tumor_lbl,is_train=True,args=args)
      
      # Test iterator
      # iter_dataset_inst_trn_tumor=iter(dataset_inst_trn_tumor)
      # trn=next(iter_dataset_inst_trn_tumor)
      # print("trn",trn)
      # ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/4036b2e1e551e14b88f7f9ada275935ec4b5bdcc.tif\n', 
      #  ['4036b2e1e551e14b88f7f9ada275935ec4b5bdcc', 0])

      # ================================================================================
      # c dataloader_tumor_trn: create dataloader
      dataloader_tumor_trn=torch.utils.data.DataLoader(
        dataset=dataset_inst_trn_tumor,batch_size=batch_size,shuffle=False,num_workers=3)
      
      # c dataloader_tumor_trn_iter: iterator of dataloader
      # dataloader_tumor_trn_iter=iter(dataloader_tumor_trn)
      
      # Test dataloader
      # pairs=next(dataloader_tumor_trn_iter)
      # print("pairs",pairs)
      # [('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/45da9ea0958e83c3a5d6efe1e9912e80ab204b59.tif\n', 
      #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/9f49570e69b43a10e58566c5207f87c4a96f04d2.tif\n'),
      #  [('45da9ea0958e83c3a5d6efe1e9912e80ab204b59', 
      #    '9f49570e69b43a10e58566c5207f87c4a96f04d2'), 
      #   tensor([1, 0])]]

      # ================================================================================
      # c num_imgs_trn: number of train image
      num_imgs_trn=len(dataset_inst_trn_tumor)
      # print("num_imgs_trn",num_imgs_trn)
      # 198022

      args.__setattr__("num_imgs_trn",num_imgs_trn)
      # print("args",args)
      # print("Current batch size:",batch_size)
      # print("Possible batch size:",list(utils_common.divisorGenerator(num_imgs_trn)))
      
      # assert str(num_imgs_trn/batch_size).split(".")[-1]==str(0),"Check batch size, currently it's incorrect"

      # ================================================================================
      # @ If you don't use Augmentor
      if args.use_augmentor=="False":
        pass      

      else: # @ If you use Augmentor

        # @ Iterate all images in dataset during single epoch
        for idx,data in enumerate(dataloader_tumor_trn):
          # c paths_of_imgs: paths of images
          paths_of_imgs=data[0]
          # print("paths_of_imgs",paths_of_imgs)

          # c labels_in_scalar: label values of images
          labels_in_scalar=np.array(data[1][1])
          # print("labels_in_scalar",labels_in_scalar)
          # labels_in_scalar [1 0 1 1 0 0 0 1 0 1 0 1 0 0 0 1 0 1 0 1 0 1 0 1 0 0 1 0 1 0 0 1 1 0 0 0 1

          # c bs_pa_tumor_d: batchsized paths of tumor dataset
          bs_pa_tumor_d=[paths_of_imgs,labels_in_scalar]

          # ================================================================================
          # @ Perform data augmentation

          # c sampled_trn_imgs_tc: sampled train images in torch tensor
          # c label_values: corresponding label values
          sampled_trn_imgs_tc,label_values=utils_data.use_augmetor_for_tumor_data(bs_pa_tumor_d,args)
          # print("sampled_trn_imgs_tc",sampled_trn_imgs_tc.shape)
          # (10, 3, 48, 48)
          # print("label_values",label_values)
          # [1, 1, 0, 0, 0, 1, 0, 0, 0, 0]

          # ================================================================================
          trn_imgs_tcv=utils_pytorch.get_dense_data_Variable(sampled_trn_imgs_tc)
          # print("trn_imgs_tcv",trn_imgs_tcv.shape)
          # torch.Size([10, 3, 48, 48])

          # ================================================================================
          # @ Remove existing gradients
          model_api_instance.remove_existing_gradients_before_starting_new_training()
          
          # ================================================================================
          # @ c predicted_labels: pass input images and get predictions
          predicted_labels=model_api_instance.gen_net(trn_imgs_tcv)
          # print("predicted_labels",predicted_labels)
          # tensor([[-0.0724],
          #         [-0.0299],
          #         [-0.1650],
          #         [-0.2458],
          #         [-0.3437],
          #         [-0.1207],
          #         [-0.3087],
          #         [-0.2381],
          #         [ 0.0811],
          #         [-0.2436]], device='cuda:0', grad_fn=<AddmmBackward>)
          
          label_tc=Variable(torch.tensor(label_values,device=predicted_labels.device))

          # ================================================================================
          # @ Calculate loss values

          criterion=nn.BCEWithLogitsLoss()

          loss_val=criterion(predicted_labels.squeeze(),label_tc)
          # print("loss_val",loss_val)

          loss_val=10.0*loss_val

          # ================================================================================
          # @ When you use 2 feature output and Cross Entropy loss function

          # # c m: LogSoftmax layer
          # m=nn.LogSoftmax()
          # # c loss: NLLLoss layer
          # loss=nn.NLLLoss()
          # # c loss_val: calculated loss value
          # loss_val=loss(m(predicted_labels),Variable(torch.Tensor(label_values).long().cuda()))
                    
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
          
          tumor_loss_temp.append(loss_val.item())

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
    plt.plot(tumor_loss_temp)
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
      dataset_inst_test_tumor=dataset_tumor.Dataset_Tumor(
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
      
      sub_ds=dataset_tumor_test.Dataset_Tumor_Submission()
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
