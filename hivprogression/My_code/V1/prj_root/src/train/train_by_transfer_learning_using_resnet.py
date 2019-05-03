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
import scipy.optimize as opt
import scipy.special
from skimage.viewer import ImageViewer
from random import shuffle
import gc
import Augmentor
import traceback
import subprocess
import csv

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
  k_fold=3
  epoch=int(args.epoch)
  batch_size=int(args.batch_size)
  # print("epoch",epoch)
  # print("batch_size",batch_size)
  # 9
  # 2

  # ================================================================================
  text_file_instance=text_file_path_api_module.Path_Of_Text_Files(args)

  txt_of_train_data=text_file_instance.train_data
  # print("txt_of_train_data",txt_of_train_data)
  # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train_csv_path.txt

  # ================================================================================
  contents_of_txt,num_line=utils_common.return_path_list_from_txt(txt_of_train_data)
  # print("contents_of_txt",contents_of_txt)
  # ['/mnt/1T-5e7/mycodehtml/prac_data_science/kaggle/hivprogression/My_code/Data/training_data.csv']
  
  # ================================================================================
  train_data_df=pd.read_csv(contents_of_txt[0],encoding='utf8')

  train_data_df=train_data_df.dropna()
  # print("train_data_df",train_data_df.shape)
  # (920, 6)

  train_data_wo_id_df=train_data_df.iloc[:,1:]
  # print("train_data_wo_id_df",train_data_wo_id_df.shape)
  # (920, 5)

  # ================================================================================
  train_k,vali_k=utils_data.get_k_folds(train_data_wo_id_df)

  # ================================================================================
  # c loss_list: list which will stores loss values to plot loss
  loss_list=[]
  f1_score_list=[]

  # ================================================================================
  # c model_api_instance: instance of model API
  model_api_instance=model_api_module.Model_API_class(args)
  # print("model_api_instance",model_api_instance)
  # <src.api_model.model_api_module.Model_API_class object at 0x7fb305557b00>

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
    for one_k in range(k_fold):
      single_train_k=train_k[one_k]
      single_vali_k=vali_k[one_k]
      single_train_lbl_k=train_lbl_k[one_k]
      single_vali_lbl_k=vali_lbl_k[one_k]

      # ================================================================================
      # @ Validation dataset
      dataset_inst_vali=custom_ds.Custom_DS_vali(single_vali_k,single_vali_lbl_k,args=args)

      dataloader_vali=torch.utils.data.DataLoader(
          dataset=dataset_inst_vali,batch_size=batch_size,shuffle=False,num_workers=3)

      for one_ep in range(epoch): # @ Iterates all epochs
        # print("single_train_k",len(single_train_k))
        # 20714
        # print("single_vali_k",len(single_vali_k))
        # 10358
        # print("single_train_lbl_k",len(single_train_lbl_k))
        # 20714
        # print("single_vali_lbl_k",len(single_vali_lbl_k))
        # 10358

        # ================================================================================
        # c dataset_inst_trn: dataset instance of tumor
        dataset_inst_trn=custom_ds.Custom_DS(single_train_k,single_train_lbl_k,args=args)
        
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
        # 20714

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
           
            bs_pa_tumor_d=utils_data.create_batch_pair_of_paths(data,args)
            # print("bs_pa_tumor_d",bs_pa_tumor_d)
            # [[('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/292d9824-bba1-11e8-b2b9-ac1f6b6435d0_blue.png',
            #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/292d9824-bba1-11e8-b2b9-ac1f6b6435d0_green.png',
            #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/292d9824-bba1-11e8-b2b9-ac1f6b6435d0_red.png',
            #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/292d9824-bba1-11e8-b2b9-ac1f6b6435d0_yellow.png'),
            #   ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/7f1e4598-bbc0-11e8-b2bb-ac1f6b6435d0_blue.png',
            #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/7f1e4598-bbc0-11e8-b2bb-ac1f6b6435d0_green.png',
            #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/7f1e4598-bbc0-11e8-b2bb-ac1f6b6435d0_red.png',
            #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/7f1e4598-bbc0-11e8-b2bb-ac1f6b6435d0_yellow.png')],
            #  array(['3','23'],dtype='<U2')]

            # ================================================================================
            # @ Perform data augmentation

            sampled_trn_imgs,label_values=utils_data.use_augmetor_for_data(bs_pa_tumor_d,args)
            # afaf 1: sampled_trn_imgs,label_values=utils_data.use_augmetor_for_data(bs_pa_tumor_d,args)
            
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
            # print("loss_val",loss_val)
            # tensor(6.5374, device='cuda:0', grad_fn=<MeanBackward1>)
                      
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
            # print("end of single batch")

          # ================================================================================
          # print("end of all batches")

        # ================================================================================
        # @ Save model after epoch
        num_epoch="epoch_"+str(one_ep)
        model_api_instance.save_model_after_epoch(num_epoch)

        # ================================================================================
        # @ Update learning rate

        scheduler.step()
        # print("scheduler.base_lrs",scheduler.base_lrs)

        # ================================================================================
        # print("End of single epoch")

      # ================================================================================
      # print("end of all epochs")
      
      # ================================================================================
      with torch.no_grad():
        n=28
        TP=torch.tensor(np.zeros(n)).float().cuda()
        FP=torch.tensor(np.zeros(n)).float().cuda()
        FN=torch.tensor(np.zeros(n)).float().cuda()

        for idx_vali,data_vali in enumerate(dataloader_vali):
          bs_pa_tumor_d_vali=utils_data.create_batch_pair_of_paths(data_vali,args)
          # print("bs_pa_tumor_d_vali",bs_pa_tumor_d_vali)
          # [[('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/0020af02-bbba-11e8-b2ba-ac1f6b6435d0_blue.png',
          #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/0020af02-bbba-11e8-b2ba-ac1f6b6435d0_green.png',
          #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/0020af02-bbba-11e8-b2ba-ac1f6b6435d0_red.png',
          #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/0020af02-bbba-11e8-b2ba-ac1f6b6435d0_yellow.png'),
          #   ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_blue.png',
          #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_green.png',
          #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_red.png',
          #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_yellow.png')],
          #  array(['25 2','16 0'],dtype='<U4')]

          img_paths=bs_pa_tumor_d_vali[0]
          labels=bs_pa_tumor_d_vali[1]
          # print("labels",labels)
          # labels ['2 0' '1']
          # print("labels",labels.shape)

          labels=[one_protein_lbl.strip().split(" ") for one_protein_lbl in labels]
          # [['5'], ['0'], ['25'], ['2'], ['23'], ['25', '4'], ['12'], ['22', '2'], ['3'], ['0', '21'], ['2'], ['25', '18', '3', '0'], ['5'], ['2', '0', '21'], ['0', '21'], ['25'], ['25'], ['23'], ['23', '0'], ['25', '2', '0']]
          # print("labels",labels)

          labels_oh=utils_common.one_hot_label_vali(batch_size,labels)
          # print("labels_oh",labels_oh)
          # [[1. 1. 1. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
          #  [0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
          labels_oh_np=np.array(labels_oh)
          labels_oh_tc=torch.tensor(labels_oh_np).cuda()

          all_images_vali_stacked=utils_data.get_batch_vali_imgs(img_paths)
          # print("all_images_vali_stacked",all_images_vali_stacked.shape)
          # (2, 4, 224, 224)

          all_images_vali_stacked_tc=utils_pytorch.get_Variable(all_images_vali_stacked)
          # print("all_images_vali_stacked_tc",all_images_vali_stacked_tc.shape)
          # torch.Size([2, 4, 224, 224])

          model_eval=model_api_instance.gen_net.eval()
          pred_vali=model_eval(all_images_vali_stacked_tc)
          # print("pred_vali",pred_vali)
          # print("pred_vali",pred_vali.shape)
          # torch.Size([2, 28])
      
          # ================================================================================
          single_TP,single_FP,single_FN=metrics_module.calculate_f1_score(pred_vali,labels_oh_tc)
          TP+=single_TP
          FP+=single_FP
          FN+=single_FN

        score=(2.0*TP/(2.0*TP+FP+FN+1e-6)).mean()
        print("score",score)
        f1_score_list.append(score.item())
        # tensor(0.0238, device='cuda:0')

    # ================================================================================
    # @ Plot loss value
    plt.plot(loss_list)
    plt.title("Loss value: 1st fold, 2nd fold, 3rd fold, continuously")
    plt.savefig("loss.png")
    plt.show()

    plt.plot(f1_score_list)
    plt.title("F1 score: 1st fold, 2nd fold, 3rd fold, continuously")
    plt.savefig("f1_score.png")
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
