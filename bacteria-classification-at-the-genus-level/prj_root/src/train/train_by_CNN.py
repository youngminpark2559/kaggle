import numpy as np
np.set_printoptions(threshold=np.nan)
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
from sklearn.model_selection import train_test_split
import scipy.misc
import scipy.optimize as opt
import scipy.special
from sklearn import svm
from skimage.viewer import ImageViewer
from random import shuffle
import gc
import Augmentor
import traceback
import subprocess
import csv
from xgboost import XGBClassifier

# ================================================================================
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
from src.metrics import metrics_module as metrics_module

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

  txt_of_image_data=text_file_instance.train_data
  # print("txt_of_image_data",txt_of_image_data)
  # /mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level/trn_imgs_paths.txt

  txt_of_label_data=text_file_instance.label_data
  # print("txt_of_label_data",txt_of_label_data)
  # /mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level/trn_imgs_labels.csv

  # ================================================================================
  model_api_instance=model_api_module.Model_API_class(args)

  path_of_imgs_tumor=[]
  with open(txt_of_image_data) as f:
    lines=f.readlines()
    path_of_imgs_tumor.extend(lines)
  # print("path_of_imgs_tumor",path_of_imgs_tumor)
  # ['/mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level/Train_Images/0004.png\n', '/mnt/1T-5e7/mycodehtml/

  pre_sale=pd.read_csv(txt_of_label_data,encoding='utf8')
  # print("pre_sale",pre_sale)
  #      Image ID          Class
  # 0    4         ecoli        
  # 1    5         salmonella   

  bacteria_label_np=np.array(pre_sale.iloc[:,1])
  # print("bacteria_label_np",bacteria_label_np)
  # ['ecoli' 'salmonella' 'staphylococus' 'listeria' 'ecoli' 'ecoli'

  zipped=list(zip(path_of_imgs_tumor,bacteria_label_np))
  # print("zipped",zipped)
  # zipped [('/mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level/Train_Images/0004.png\n', array([4, 'ecoli'], dtype=object)), ('/mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level/Train_Images/0005.png\n', array([5, 
  # [('/mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level/Train_Images/0004.png\n', 'ecoli'), ('/mnt/1T-5e7/

  shuffle(zipped)

  zipped_np=np.array(zipped)
  X=zipped_np[:,0]
  y=zipped_np[:,1]

  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.10,random_state=42)
  # print("X_train",np.array(X_train).shape)
  # # (545,)
  # print("y_train",np.array(y_train).shape)
  # # (545,)
  # print("X_test",np.array(X_test).shape)
  # # (61,)
  # print("y_test",np.array(y_test).shape)
  # # (61,)

  # ================================================================================
  # @ Test Grad CAM
  # imgs=["/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/33/82d4d190d2fed1be255fc3bac36a37c860bb31c0.tif",
  #       "/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/33/82a5300cd61628fb9bae332cdb7d5e7e37b1fb36.tif"]
  # grad_cam.initialize_grad_cam(model=model_api_instance.gen_net,list_of_img_paths=imgs,args=args)
  # afaf 1: grad_cam.initialize_grad_cam(model=model_api_instance.gen_net,list_of_img_paths=imgs,args=args)

  tumor_loss_temp=[]
  # ================================================================================
  if args.task_mode=="train": # If you're in train mode
    
    # ================================================================================
    # @ Configure learning rate scheduler

    # Update learning rate 4 times during entire epochs
    # For example, if you use 10 epochs, int(10/4), 1 2 / 3 4 / 5 6 / 7 8 / 9 10
    # 0-1 epochs: 0.001 -> 2-3 epochs: 0.0001 -> 4-5 epochs: 0.00001 -> 5-6 epochs: 0.000001

    # scheduler=StepLR(model_api_instance.optimizer,step_size=int(epoch/4),gamma=0.1)

    # ================================================================================
    for one_ep in range(epoch): # @ Iterates all epochs

      # c dataset_inst_trn_tumor: dataset instance of tumor
      dataset_inst_trn_tumor=custom_ds.Custom_DS(single_train_k=X_train,single_train_lbl_k=y_train,args=args)

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
          # print("data",data)
          # [('/mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level/Train_Images/1611.png\n', 
          #   '/mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level/Train_Images/1475.png\n'), 
          #  ('staphylococus', 'staphylococus')]

          # c paths_of_imgs: paths of images
          paths_of_imgs=data[0]
          # print("paths_of_imgs",paths_of_imgs)
          # ('/mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level/Train_Images/0849.png\n', '/mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level/Train_Images/0969.png\n')

          # c labels_in_scalar: label values of images
          labels_in_scalar=np.array(data[1])
          # print("labels_in_scalar",labels_in_scalar)
          # ['ecoli' 'salmonella']

          # c bs_pa_tumor_d: batchsized paths of tumor dataset
          bs_pa_tumor_d=[paths_of_imgs,labels_in_scalar]

          # ================================================================================
          # @ Perform data augmentation

          # c sampled_trn_imgs_tc: sampled train images in torch tensor
          # c label_values: corresponding label values
          sampled_trn_imgs_tc,label_values=utils_data.use_augmetor_for_tumor_data(bs_pa_tumor_d,args)
          # print("sampled_trn_imgs_tc",sampled_trn_imgs_tc.shape)
          # (2, 1, 224, 224)

          # print("label_values",label_values)
          # ['ecoli', 'staphylococus']


          I_4x4=np.eye(4)
          # print("I_4x4",I_4x4)
          # [[1. 0. 0. 0.]
          #  [0. 1. 0. 0.]
          #  [0. 0. 1. 0.]
          #  [0. 0. 0. 1.]]
          bac_name_dict={'ecoli':[1.,0.,0.,0.,],'salmonella':[0.,1.,0.,0.],'staphylococus':[0.,0.,1.,0.],'listeria':[0.,0.,0.,1.]}

          one_hot_label=[]
          for one_bac_name in label_values:
            # print("one_bac_name",one_bac_name)
            # listeria
            
            one_hot=bac_name_dict[one_bac_name]
            # print("one_hot",one_hot)
            # [0.0, 0.0, 0.0, 1.0]

            one_hot_label.append(one_hot)

          # print("one_hot_label",one_hot_label)
          # [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]

          # bac_name_dict={'ecoli':1,'salmonella':2,'staphylococus':3,'listeria':4}

          # one_hot_label=[]
          # for one_bac_name in label_values:
          #   # print("one_bac_name",one_bac_name)
          #   # listeria
            
          #   one_hot=bac_name_dict[one_bac_name]
          #   # print("one_hot",one_hot)
          #   # [0.0, 0.0, 0.0, 1.0]

          #   one_hot_label.append(one_hot)

          # # print("one_hot_label",one_hot_label)
          # # [3, 3]




          y1 = torch.FloatTensor(one_hot_label)
          _, targets = y1.max(dim=1)
          # print("targets",targets)
          # afaf

          # ================================================================================
          trn_imgs_tcv=utils_pytorch.get_Variable(sampled_trn_imgs_tc)
          # print("trn_imgs_tcv",trn_imgs_tcv.shape)
          # torch.Size([2, 1, 224, 224])

          # ================================================================================
          # @ Remove existing gradients
          model_api_instance.remove_existing_gradients_before_starting_new_training()
          
          # ================================================================================
          # @ c predicted_labels: pass input images and get predictions
          predicted_labels=model_api_instance.gen_net(trn_imgs_tcv)
          # print("predicted_labels",predicted_labels)
          # tensor([[-0.1724, -0.1855, -0.1478,  0.1190],
          #         [-0.1837, -0.1834, -0.1849,  0.0015]], device='cuda:0',
          #       grad_fn=<AddmmBackward>)

          # max_indics=torch.argmax(predicted_labels,dim=1)
          # # print("max_indics",max_indics)
          # # tensor([3, 3], device='cuda:0')
          # print("max_indics",max_indics.shape)
          # afaf
          
          label_tc=Variable(torch.tensor(targets,device=predicted_labels.device))
          # print("label_tc",label_tc)
          # tensor([[0., 1., 0., 0.],
          #         [0., 1., 0., 0.]], device='cuda:0')

          # ================================================================================
          # @ Calculate loss values

          criterion=nn.CrossEntropyLoss()

          # print("max_indics",max_indics)
          # tensor([2, 2], device='cuda:0')

          # print("label_tc",label_tc)
          # tensor([2, 1], device='cuda:0')

          print("predicted_labels",predicted_labels.shape)
          # print("label_tc",label_tc.shape)
          # predicted_labels torch.Size([2, 4])
          # label_tc torch.Size([2])

          loss_val=criterion(predicted_labels,label_tc.long())
          # print("loss_val",loss_val)
          # tensor(1.7040, device='cuda:0', grad_fn=<NllLossBackward>)

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

      # scheduler.step()
      # print("scheduler.base_lrs",scheduler.base_lrs)

    # ================================================================================
    # @ Plot loss value
    plt.plot(tumor_loss_temp)
    plt.savefig("loss.png")
    plt.show()

    with torch.no_grad():

      dataset_inst_vali_tumor=custom_ds.Custom_DS_vali(single_vali_k=X_test,single_vali_lbl_k=y_test,args=args)

      dataloader_tumor_vali=torch.utils.data.DataLoader(
        dataset=dataset_inst_vali_tumor,batch_size=5,shuffle=False,num_workers=3)

      # ================================================================================
      num_imgs_trn=len(dataset_inst_vali_tumor)
      # print("num_imgs_trn",num_imgs_trn)
      # 198022

      pred_np_li=[]
      label_np_li=[]

      # @ Iterate all images in dataset during single epoch
      for idx,data in enumerate(dataloader_tumor_vali):
        # print("data",data)
        # [('/mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level/Train_Images/1611.png\n', 
        #   '/mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level/Train_Images/1475.png\n'), 
        #  ('staphylococus', 'staphylococus')]

        # c paths_of_imgs: paths of images
        paths_of_imgs=data[0]
        # print("paths_of_imgs",paths_of_imgs)
        # ('/mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level/Train_Images/0849.png\n', '/mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level/Train_Images/0969.png\n')

        intput_imgs=[]
        for one_path in paths_of_imgs:
          loaded_img=utils_image.load_img(one_path.replace("\n",""))
          # print("loaded_img",loaded_img.shape)
          # (1024, 1280)

          loaded_img=utils_image.resize_img(loaded_img,224,224)
          intput_imgs.append(loaded_img)

        intput_imgs_np=np.array(intput_imgs)
        # print("intput_imgs_np",intput_imgs_np.shape)
        # (2, 224, 224)

        intput_imgs_np=intput_imgs_np[:,np.newaxis,:,:]
        intput_imgs_tc=torch.tensor(intput_imgs_np).float()

        trn_imgs_tcv=utils_pytorch.get_Variable(intput_imgs_tc)
        # print("trn_imgs_tcv",trn_imgs_tcv.shape)
        # torch.Size([2, 1, 224, 224])

        # c labels_in_scalar: label values of images
        labels_in_scalar=np.array(data[1])
        # print("labels_in_scalar",labels_in_scalar)
        # ['salmonella' 'listeria' 'salmonella' 'salmonella' 'salmonella']
        
        I_4x4=np.eye(4)
        # print("I_4x4",I_4x4)
        # [[1. 0. 0. 0.]
        #  [0. 1. 0. 0.]
        #  [0. 0. 1. 0.]
        #  [0. 0. 0. 1.]]
        bac_name_dict={'ecoli':[1.,0.,0.,0.,],'salmonella':[0.,1.,0.,0.],'staphylococus':[0.,0.,1.,0.],'listeria':[0.,0.,0.,1.]}

        one_hot_label=[]
        for one_bac_name in labels_in_scalar:
          # print("one_bac_name",one_bac_name)
          # salmonella
          
          one_hot=bac_name_dict[one_bac_name]
          # print("one_hot",one_hot)
          # [1.0, 0.0, 0.0, 0.0]

          one_hot_label.append(one_hot)

        # print("one_hot_label",one_hot_label)
        # [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]

        y1 = torch.FloatTensor(one_hot_label)
        _, targets = y1.max(dim=1)
        # print("targets",targets)
        # tensor([1, 0, 1, 1, 3])

        # ================================================================================
        # @ Remove existing gradients
        # model_api_instance.remove_existing_gradients_before_starting_new_training()
        
        # ================================================================================
        # @ c predicted_labels: pass input images and get predictions
        predicted_labels=model_api_instance.gen_net(trn_imgs_tcv)
        # print("predicted_labels",predicted_labels)
        # tensor([[ 0.6513, -0.1330, -0.5001, -0.0210],
        #         [ 0.1990,  0.1838, -0.1278, -0.1546]], device='cuda:0')

        max_indics=torch.argmax(predicted_labels,dim=1)
        # print("max_indics",max_indics)
        # tensor([0, 0], device='cuda:0')

        # print("targets",targets)
        # tensor([0, 1])
        
        
        label_tc=Variable(torch.tensor(targets,device=predicted_labels.device))
        # print("label_tc",label_tc)
        # tensor([1, 3, 2, 0, 2], device='cuda:0')

        pred_np=max_indics.detach().cpu().numpy()
        label_np=label_tc.detach().cpu().numpy()

        pred_np_li.append(pred_np)
        label_np_li.append(label_np)

      # pred_np_li [array([0, 1])]
      # label_np_li [array([3, 1])]


      label_np_li_len=len(label_np_li)

      y_true=np.array(label_np_li)
      y_pred=np.array(pred_np_li)
      print("y_true",y_true)
      # y_true [
      # array([2, 1, 2, 2, 1]) array([0, 2, 2, 3, 2]) array([1, 1, 0, 0, 2])
      # array([1, 0, 1, 3, 2]) array([2, 3, 3, 0, 0]) array([0, 3, 3, 3, 1])
      # array([3, 2, 3, 2, 1]) array([1, 2, 3, 0, 2]) array([1, 1, 2, 3, 3])
      # array([1, 3, 3, 0, 3]) array([0, 1, 0, 3, 2]) array([3, 3, 3, 3, 1])
      # array([1])]

      print("y_pred",y_pred)
      # y_pred [
      # array([2, 0, 2, 2, 0]) array([0, 2, 2, 3, 2]) array([1, 1, 0, 0, 2])
      # array([1, 0, 1, 3, 2]) array([2, 3, 3, 0, 0]) array([0, 3, 3, 3, 1])
      # array([3, 2, 3, 2, 1]) array([1, 2, 3, 0, 2]) array([1, 1, 2, 3, 3])
      # array([1, 3, 2, 0, 3]) array([0, 1, 0, 3, 2]) array([3, 2, 3, 3, 1])
      # array([0])]

      num_entire_sam=0
      num_matched=0
      for one_pair in list(zip(y_true,y_pred)):
        y_one_true=one_pair[0]
        y_one_pred=one_pair[1]

        num_sample=y_one_true.reshape(-1).shape[0]
        num_entire_sam=num_sample+num_entire_sam

        y_one_true=np.array(y_one_true)
        y_one_pred=np.array(y_one_pred)

        matched=(y_one_true==y_one_pred)

        matched_num=matched.sum()
        num_matched=num_matched+matched_num
      acc=num_matched/num_entire_sam
      print("acc",acc)
      afaf


      y_true_rs=np.reshape(y_true,(-1))
      print("y_true_rs",y_true_rs.shape)
      # (10,)
      num_sam=y_true_rs.shape[0]
      print("num_sam",num_sam)

      matched=np.array(y_true==y_pred)
      print("matched",matched)
      # [[False  True]]

      matched_num=matched.sum()
      print("matched_num",matched_num)
      # 1

      print("label_np_li_len",label_np_li_len)
      acc=matched_num/num_sam
      print("acc",acc)
      afaf





      # zpped=list(zip(y_true,y_pred))
      # n_classes = 4
      # y_true_oh=[]
      # y_pred_oh=[]
      # for o_true,o_pred in zpped:
      #   aa=np.eye(n_classes)[o_true]
      #   bb=np.eye(n_classes)[o_pred]
      #   y_true_oh.append(list(aa))
      #   y_pred_oh.append(list(bb))
      
      # # print("y_true_oh",y_true_oh)
      # # print("y_pred_oh",y_pred_oh)
      # # [array([[0., 0., 0., 1.],
      # #         [1., 0., 0., 0.]])]
      # # [array([[0., 0., 1., 0.],
      # #         [0., 0., 1., 0.]])]
      
      # y_true=np.array(y_true_oh)
      # y_pred=np.array(y_pred_oh)
      # # print("y_true",y_true)
      # # print("y_pred",y_pred)
      # # print("y_true",type(y_true))
      # # print("y_pred",type(y_pred))

      # ================================================================================
      # @ Binary Confusion Matrix

      # b_c_mat=confusion_matrix(y_true,y_pred,labels=[0,1,2,3])
      # print("b_c_mat",b_c_mat)
      # [[30  2]
      #  [ 0 68]]

      # True Positive (Tumor pic is predicted as tumor)      False Negative (Tumor pic is predicted as non-tumor)
      # False Positive (Non-tumor pic is predicted as tumor) True Negative (Non-tumor pic is predicted as non-tumor)
      
      # ================================================================================
      # @ metric report

      # report=classification_report(y_true,y_pred,target_names=['class Non tumor (neg)', 'class Tumor (pos)'])
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
      afaf

      

  
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
