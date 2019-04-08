import numpy as np
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

from src.model_api import model_api_module as model_api_module

from src.text_file_path_api import text_file_path_api_module as text_file_path_api_module

# ================================================================================
def train(args):
  wd=0
  epoch=int(args.epoch)
  batch_size=int(args.batch_size)
  # print("epoch",epoch)
  # print("batch_size",batch_size)
  # 200
  # 22

  text_file_instance=text_file_path_api_module.Path_Of_Text_Files(args)
  tumor_trn=text_file_instance.tumor_trn
  tumor_lbl=text_file_instance.tumor_lbl
  # print("tumor_trn",tumor_trn)
  # print("tumor_lbl",tumor_lbl)
  # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/trn_txt_file_processed.txt
  # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_labels.csv

  # c tumor_loss_temp: list which will stores loss values to plot loss
  tumor_loss_temp=[]
  # If you're in train mode
  if args.train_mode=="True":
    # c model_api_instance: instance of model API
    model_api_instance=model_api_module.Model_API_class(args)
    
    # ================================================================================
    # @ Configure learning rate scheduler
    # Update learning rate 4 times during entire epochs
    # For example, if you use 10 epochs, int(10/4), 1 2 / 3 4 / 5 6 / 7 8 / 9 10
    # 0-1 epochs: 0.001 -> 2-3 epochs: 0.0001 -> 4-5 epochs: 0.00001 -> 5-6 epochs: 0.000001

    scheduler=StepLR(model_api_instance.optimizer,step_size=int(epoch/4),gamma=0.1)

    # ================================================================================
    # @ Test Grad CAM
    # imgs=["/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/test/ffcaef8b9006b4d0b128328e6df6e4d139d3c40a.tif",
    #       "/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/test/ffcc29cf0e363737b577d1db470df0bb1adf7957.tif"]
    # grad_cam.initialize_grad_cam(
    #   model=gen_net,
    #   list_of_img_paths=imgs,
    #   args=args)

    # ================================================================================
    # @ Iterates all epochs
    for one_ep in range(epoch):
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

      # @ If you dont' use Augmentor
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
          sampled_trn_imgs_tc,label_values=\
            utils_data.use_augmetor_for_tumor_data(bs_pa_tumor_d,args)
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
          # @ For adamW
          if args.optimizer=="adamW":
            # the block below changes weight decay in adam
            for group in model_api_instance.optimizer.param_groups:
              for param in group['params']:
                param.data=param.data.add(-wd*group['lr'],param.data)
          
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
  else: # @ Test the trained model
    with torch.no_grad(): # @ Use network without calculating gradients
      # tumor_trn=args.dir_where_text_file_for_image_paths_is_in+"/tumor_trn.txt"
      # tumor_lbl=args.dir_where_text_file_for_image_paths_is_in+"/train_labels.csv"
      # print("tumor_trn",tumor_trn)
      # print("tumor_lbl",tumor_lbl)
      # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/tumor_trn.txt
      # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_labels.csv

      # ================================================================================
      # @ c dataset_inst_test_tumor: dataset instance of tumor dataset
      dataset_inst_test_tumor=dataset_tumor.Dataset_Tumor(
        txt_containing_paths=tumor_trn,txt_containing_labels=tumor_lbl,is_train=False,args=args)

      # ================================================================================
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
      # @ c entire_correct_cases: placeholder for entire correct cases
      entire_correct_cases=0.0
      predicted_values=[]
      true_values=[]
      img_paths=[]

      # @ Iterate all batches (batch1+batch2+...+batchn=entire images)
      for idx,data in enumerate(dataloader_tumor_test):
        # print("idx",idx)
        # print("data",data)
        # [('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/e693f9ac4097289c317831960514b78701999cd9.tif\n',
        #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/e6941f6c6825e7c409b9364e2fb6c2d629df8a76.tif\n',),
        #  [('e693f9ac4097289c317831960514b78701999cd9','e6941f6c6825e7c409b9364e2fb6c2d629df8a76'),tensor([1,0])]]
        
        # ================================================================================
        # @ c imgs: validation images
        imgs=data[0]

        img_paths.extend(imgs)

        # @ c imgs: label to validation images
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
        #         [-0.5436],
        #         [-2.3111],
        #         [-2.6922],
        #         [-0.1253],
        #         [-2.5247],
        #         [-1.0271],
        #         [ 3.7961],
        #         [-2.2452],
        #         [-1.9822],
        #         [-0.5083],
        #         [-0.3109],
        #         [-0.0466],
        #         [ 3.0260],
        #         [-1.9039],
        #         [-1.0872],
        #         [-0.9267],
        #         [-1.0542],
        #         [-2.7592],
        #         [-1.6940],
        #         [-0.7462],
        #         [-0.6184],
        #         [ 0.6964],
        #         [ 4.2202],
        #         [-2.8205],
        #         [ 5.1924],
        #         [ 2.5785],
        #         [-1.3629],
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
        
        true_values.extend(lbls)
      
      # ================================================================================
      # print("predicted_values",predicted_values)
      # [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0]
      # print("true_values",true_values)
      # [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
      # print("img_paths",img_paths)

      # ================================================================================
      # Tumor pic: 32
      # print(np.sum((np.array(true_values)==1).astype("float32")))
      # 32.0

      # Non-tumor pic: 68
      # print(np.sum((np.array(true_values)==0).astype("float32")))
      # 68.0

      # ================================================================================
      mask_for_different=np.array(predicted_values)!=np.array(true_values)
      different_predicted_values=np.array(predicted_values)[mask_for_different]
      different_true_values=np.array(true_values)[mask_for_different]
      different_imgs=np.array(img_paths)[mask_for_different]
      different_sets=list(zip(different_predicted_values,different_true_values,different_imgs))
      # print("different_sets",different_sets)
      # [(1, 0, '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/00/010bd8192a71e611ee9617f0aabb4dc795db3159.tif\n'),
      #  (1, 0, '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/00/010d236bdd4a6c79ebc8c41dbdc1f889a0c95e12.tif\n'),
      #  (1, 0, '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/00/010dc9a11572ccfe3c9e4c6026e4da389ede5a59.tif\n'),
      #  (0, 1, '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/00/010fd1d36f89041a6ffca32e95ea70ed43211e96.tif\n'),
      #  (1, 0, '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/00/01101aa48a28b1e39f3b58dc2767832ec02a00de.tif\n'),
      #  (0, 1, '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/00/01108d2ed060b7587629a8ee3ee1069fe8c41a03.tif\n'),
      #  (0, 1, '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/00/01119759a84c3dccbcdf3fd42a76cd060fc33155.tif\n'),
      #  (0, 1, '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/00/011273075f488bf43f8f7dc7c36234919f1c2e61.tif\n'),
      #  (1, 0, '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/00/01142529dde4e0d5980e8f7d158d61f839b67c83.tif\n'),
      #  (0, 1, '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/00/0118370bbbf2251881977095e6bb99917df90e72.tif\n'),
      #  (1, 0, '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/00/0119d1d935ca4d061cd17fddd0fe7e899ebad403.tif\n'),
      #  (1, 0, '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/00/011a4ed4e8e672b75df64f27a12f7f8cdbd900ae.tif\n'),
      #  (1, 0, '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/00/011c2ff45bba6226c20d5f4b9418797d25a6847e.tif\n'),
      #  (1, 0, '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/00/011feab60278c3dde83038cfa6a1bdc3d21b04f0.tif\n'),
      #  (0, 1, '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/00/0120e3db75e9ac417d839d88fad8a581b5c82091.tif\n'),
      #  (1, 0, '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/00/012482117dc022e27926795ca96ece0fe0f6ab10.tif\n'),
      #  (0, 1, '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/00/01256d543460ac6cdff51dbb4b1dc1ceed1d3c36.tif\n'),
      #  (0, 1, '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/00/012729816c718e8c29510fbec594aa2c6d444f35.tif\n'),
      #  (1, 0, '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/00/0129e0783feef22fe2d470f5af3f2a5684d3920b.tif\n'),
      #  (1, 0, '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_split/00/012a7132573451b51bb9a5a296b70c3f16adcc67.tif\n')]

      y_true=true_values
      y_pred=predicted_values

      # ================================================================================
      # @ Binary Confusion Matrix
      # from sklearn.metrics import confusion_matrix
      # c_mat=confusion_matrix(true_values,predicted_values,labels=[1,0])
      # # print("c_mat",c_mat)
      # # True Positive (Tumor pic is predicted as tumor)      False Negative (Tumor pic is predicted as non-tumor)
      # # False Positive (Non-tumor pic is predicted as tumor) True Negative (Non-tumor pic is predicted as non-tumor)
      # # [[24  8]
      # #  [12 56]]

      # ================================================================================
      # from sklearn.metrics import classification_report
      # y_true=true_values
      # y_pred=predicted_values
      # # print(classification_report(y_true,y_pred,target_names=['class Non tumor (neg)', 'class Tumor (pos)']))
      # #                        precision    recall  f1-score   support

      # # class Non tumor (neg)       0.88      0.82      0.85        68
      # #     class Tumor (pos)       0.67      0.75      0.71        32

      # #             micro avg       0.80      0.80      0.80       100
      # #             macro avg       0.77      0.79      0.78       100
      # #          weighted avg       0.81      0.80      0.80       100

      # ================================================================================
      from sklearn.metrics import accuracy_score,precision_score,recall_score,fbeta_score,f1_score
      print("accuracy_score",accuracy_score(y_true, y_pred))
      # # 0.8

      print("precision_score",precision_score(y_true, y_pred))
      # # 0.6666666666666666

      print("recall_score",recall_score(y_true, y_pred))
      # # 0.75

      # # print("fbeta_score",fbeta_score(y_true, y_pred, beta))
      
      print("f1_score",fbeta_score(y_true, y_pred, beta=1))
      # # 0.7058823529411765

      # ================================================================================
      # @ ROC curve
      # from sklearn.metrics import roc_curve
      # y_true=true_values
      # y_pred=predicted_values
      # fpr,tpr,thresholds=roc_curve(y_true,y_pred)
      # plt.plot(fpr, tpr, 'o-', label="Logistic Regression")
      # # plt.plot([0, 1], [0, 1], 'k--', label="random guess")
      # # plt.plot([fallout], [recall], 'ro', ms=10)
      # # plt.xlabel('False Positive Rate (Fall-Out)')
      # # plt.ylabel('True Positive Rate (Recall)')
      # plt.title('Receiver operating characteristic example')
      # plt.show()

      # ================================================================================
      num_all_sample=len(y_pred)
      # print("num_all_sample",num_all_sample)
      # 100
      
      y_pred
      y_true
      
      y_pred_np=np.array(y_pred)
      y_true_np=np.array(y_true)

      mask_for_diff=y_pred_np!=y_true_np
      y_pred_diff=y_pred_np[mask_for_diff]
      y_true_diff=y_true_np[mask_for_diff]
      print(y_pred_diff)
      print(y_true_diff)
      # [1 1 1 0 1 0 0 0 1 0 1 1 1 1 0 1 0 0 1 1]
      # [0 0 0 1 0 1 1 1 0 1 0 0 0 0 1 0 1 1 0 0]

      num_of_pred_said_1_incorrectly=np.sum((y_pred_diff==1).astype("float32"))
      num_of_pred_said_0_incorrectly=np.sum((y_pred_diff==0).astype("float32"))
      print(num_of_pred_said_1_incorrectly)
      # 12.0
      print(num_of_pred_said_0_incorrectly)
      # 8.0

      mask_for_same=y_pred_np==y_true_np
      y_pred_same=y_pred_np[mask_for_same]
      y_true_same=y_true_np[mask_for_same]
      print(y_pred_same)
      print(y_true_same)
      # [0 0 0 1 0 1 1 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 1 1 0 0 1 0 0 0 1 0 1 0 1 1 1
      #  0 0 1 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 1 0 1 0 1 0 1 0 0 1 0 0
      #  0 1 0 0 0 0]
      # [0 0 0 1 0 1 1 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 1 1 0 0 1 0 0 0 1 0 1 0 1 1 1
      #  0 0 1 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 1 0 1 0 1 0 1 0 0 1 0 0
      #  0 1 0 0 0 0]
      
      num_of_pred_said_1_correctly=np.sum((y_pred_same==1).astype("float32"))
      num_of_pred_said_0_correctly=np.sum((y_pred_same==0).astype("float32"))
      print(num_of_pred_said_1_correctly)
      # 24.0
      print(num_of_pred_said_0_correctly)
      # 56.0
      
      
      acc=(num_of_pred_said_1_correctly+num_of_pred_said_0_correctly)/(num_of_pred_said_1_correctly+num_of_pred_said_0_correctly+num_of_pred_said_1_incorrectly+num_of_pred_said_0_incorrectly)
      print("acc",acc)
      # 0.8
   

      
      


      # ================================================================================
      # @ Calculate true positive
      # acc_true_positive=entire_correct_cases/num_imgs_test
      # print("acc_true_positive",acc_true_positive)

