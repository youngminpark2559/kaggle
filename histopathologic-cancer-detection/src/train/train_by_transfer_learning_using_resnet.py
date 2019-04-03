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
from skimage.viewer import ImageViewer
from random import shuffle
import gc
import Augmentor
import traceback
import subprocess
# import horovod.torch as hvd
# Pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
from torch.autograd import gradcheck
# src/utils
from src.utils import utils_common as utils_common
from src.utils import utils_image as utils_image
from src.utils import utils_net as utils_net
from src.utils import utils_pytorch as utils_pytorch
from src.utils import utils_data as utils_data
from src.utils import utils_hook_functions as utils_hook_functions
from src.utils import utils_visualize_gradients as utils_visualize_gradients
# src/utils_for_dataset
from src.utils_for_dataset import dataset_tumor as dataset_tumor
# src/model_api
from src.model_api import model_api_module as model_api_module

def train(args):
  epoch=int(args.epoch)
  batch_size=int(args.batch_size)
  # print("epoch",epoch)
  # print("batch_size",batch_size)
  # 200
  # 22
  tumor_trn=args.text_file_for_paths_dir+"/tumor_trn.txt"
  tumor_lbl=args.text_file_for_paths_dir+"/train_labels.csv"
  # print("tumor_trn",tumor_trn)
  # print("tumor_lbl",tumor_lbl)
  # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/tumor_trn.txt
  # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_labels.csv
  # c tumor_loss_temp: list which will stores loss values to plot loss
  tumor_loss_temp=[]
  # If you're in train mode
  if args.train_mode=="True":
    # c model_api_instance: instance of model API
    model_api_instance=model_api_module.Model_API_class(args)
    # @ Test Grad CAM
    # imgs=["/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/test/ffcaef8b9006b4d0b128328e6df6e4d139d3c40a.tif",
    #       "/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/test/ffcc29cf0e363737b577d1db470df0bb1adf7957.tif"]
    # grad_cam.initialize_grad_cam(
    #   model=gen_net,
    #   list_of_img_paths=imgs,
    #   args=args)
    # @ Iterates all epochs
    for one_ep in range(epoch):
      # c dataset_inst_trn_tumor: dataset instance of tumor
      dataset_inst_trn_tumor=dataset_tumor.Dataset_Tumor(
        txt_containing_paths=tumor_trn,
        txt_containing_labels=tumor_lbl,
        is_train=True,
        args=args)
      # Test iterator
      # iter_dataset_inst_trn_tumor=iter(dataset_inst_trn_tumor)
      # trn=next(iter_dataset_inst_trn_tumor)
      # print("trn",trn)
      # ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/4036b2e1e551e14b88f7f9ada275935ec4b5bdcc.tif\n', 
      #  ['4036b2e1e551e14b88f7f9ada275935ec4b5bdcc', 0])
      # c dataloader_tumor_trn: create dataloader
      dataloader_tumor_trn=torch.utils.data.DataLoader(
        dataset=dataset_inst_trn_tumor,
        batch_size=batch_size,
        shuffle=False,
        num_workers=3)
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
        # @ Iterate all images in dataset
        for idx,data in enumerate(dataloader_tumor_trn):
          # @ Print index of one batch
          # Batch1+Batch2+...+Batchn=entire dataset
          # if idx%100==0:
          #   print("idx",idx)
          # @ Check idx and data
          # print("idx",idx)
          # 0
          # print("data",data)
          # [('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/629dbe7b730b74b66b692cb18fb6cd351a5e6e2d.tif\n',
          #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/8333e02cdc858d3c842a13e8d2af72b553ba18a2.tif\n'),
          #  [('629dbe7b730b74b66b692cb18fb6cd351a5e6e2d','8333e02cdc858d3c842a13e8d2af72b553ba18a2'),tensor([0,1])]]
          # Get paths from data
          paths_of_imgs=data[0]
          # print("paths_of_imgs",paths_of_imgs)
          # ('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/c339d943a80a47a9a10557a80f74768b72cc40d8.tif\n',
          #  '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/5b6cb58c34d48ec9dc959e9dbccbe2eeb85dbfe2.tif\n',
          #  '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/a66c12df47562aa392dcbb6f5dd28c12e5a69f97.tif\n',
          #  '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/b232b30316e3f851669e71e9dc6fc22a21dd2455.tif\n',
          #  '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/7f93699ec777e2f853dcdc40d19b8aaca9606e39.tif\n')
          # @ Get labels from data
          labels_in_scalar=np.array(data[1][1])
          # print("data[1][1]",np.array(data[1][1]))
          # [0 1 0 1 0]
          # c tumor_d_bs_pa: tumor dataset batchsized paths
          tumor_d_bs_pa=[paths_of_imgs,labels_in_scalar]
          # @ Perform data augmentation
          # c sampled_trn_imgs_tc: sampled train images in torch tensor
          # c label_values: corresponding label values
          sampled_trn_imgs_tc,label_values=utils_data.use_augmetor_for_tumor_data(
              tumor_d_bs_pa,args)
          # print("sampled_trn_imgs_tc",sampled_trn_imgs_tc.shape)
          # print("label_values",label_values)
          # (5, 3, 64, 64)
          # [1, 1, 1, 1, 0]
          # @ Convert torch tensor into torch Variables
          # c trn_imgs_tcv: train images torch Variable
          trn_imgs_tcv=utils_pytorch.get_dense_data_Variable(sampled_trn_imgs_tc)
          # print("trn_imgs_tcv",trn_imgs_tcv.shape)
          # torch.Size([5, 3, 64, 64])
          # @ Remove existing gradients
          model_api_instance.remove_existing_gradients_before_starting_new_training()
          # @ c predicted_labels: pass input images and get predictions
          predicted_labels=model_api_instance.gen_net(trn_imgs_tcv)
          # print("predicted_labels",predicted_labels)
          # tensor([[0.6330, 0.3670],
          #         [0.7072, 0.2928],
          #         [0.5223, 0.4777],
          #         [0.4209, 0.5791],
          #         [0.5027, 0.4973],
          #         [0.5223, 0.4777],
          #         [0.5556, 0.4444],
          #         [0.5022, 0.4978],
          #         [0.5005, 0.4995],
          #         [0.5067, 0.4933],
          #         [0.3643, 0.6357]], device='cuda:0', grad_fn=<SoftmaxBackward>)
          # @ Calculate loss values
          # c m: LogSoftmax layer
          m=nn.LogSoftmax()
          # c loss: NLLLoss layer
          loss=nn.NLLLoss()
          # c loss_val: calculated loss value
          loss_val=loss(m(predicted_labels),Variable(torch.Tensor(label_values).long().cuda()))
          # @ Calculate gradient values through backpropagation
          loss_val.backward()
          # @ Update parameters of the network based on gradients
          model_api_instance.optimizer.step()
          # @ If you want to print loss
          if args.use_loss_display=="True":
            print("loss_from_one_batch",loss_from_one_batch.item())
          tumor_loss_temp.append(loss_val.item())
      # @ Save model after epoch
      model_api_instance.save_model_after_epoch(one_ep)
    # @ Plot loss value
    plt.plot(tumor_loss_temp)
    plt.savefig("loss.png")
    plt.show()
  else: # @ Test the trained model
    with torch.no_grad(): # @ Use network without calculating gradients
      tumor_trn=args.text_file_for_paths_dir+"/tumor_trn.txt"
      tumor_lbl=args.text_file_for_paths_dir+"/train_labels.csv"
      # print("tumor_trn",tumor_trn)
      # print("tumor_lbl",tumor_lbl)
      # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/tumor_trn.txt
      # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train_labels.csv
      # @ c dataset_inst_test_tumor: dataset instance of tumor dataset
      dataset_inst_test_tumor=dataset_tumor.Dataset_Tumor(
        txt_containing_paths=tumor_trn,
        txt_containing_labels=tumor_lbl,
        is_train=False,
        args=args)
      # @ c dataloader_tumor_test: dataloader instance of tumor dataset
      dataloader_tumor_test=torch.utils.data.DataLoader(
        dataset=dataset_inst_test_tumor,
        batch_size=batch_size,
        shuffle=False,
        num_workers=3)
      # @ c num_imgs_test: number of entire test images
      num_imgs_test=len(dataset_inst_test_tumor)
      # @ Create network and optimizer
      if args.train_method=="train_by_transfer_learning_using_resnet":
        model_api_instance=model_api_module.Model_API_class(args)
      # @ c entire_correct_cases: placeholder for entire correct cases
      entire_correct_cases=0.0
      # @ Iterate all batches (batch1+batch2+...+batchn=entire images)
      for idx,data in enumerate(dataloader_tumor_test):
        # print("idx",idx)
        # print("data",data)
        # [('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/e693f9ac4097289c317831960514b78701999cd9.tif\n',
        #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/train/e6941f6c6825e7c409b9364e2fb6c2d629df8a76.tif\n',),
        #  [('e693f9ac4097289c317831960514b78701999cd9','e6941f6c6825e7c409b9364e2fb6c2d629df8a76'),tensor([1,0])]]
        # @ c imgs: validation images
        imgs=data[0]
        # @ c imgs: label to validation images
        lbls=data[1][1].numpy()
        # @ c num_imgs: number of validation image in one batch
        num_imgs=lbls.shape[0]
        # print("num_imgs",num_imgs)
        # 11
        # @ Load images from paths
        test_imgs_list=[]
        for one_img_path in imgs:
          one_loaded_img=utils_image.load_img(one_img_path)
          # print("one_loaded_img",one_loaded_img.shape)
          # (96, 96, 3)
          test_imgs_list.append(one_loaded_img)
        test_imgs_np=np.array(test_imgs_list).transpose(0,3,1,2)
        # @ If you want to use center (48,48) image from (96,96) image
        # test_imgs_np=test_imgs_np[:,:,24:72,24:72]
        # print("test_imgs_np",test_imgs_np.shape)
        # (11, 3, 48, 48)
        test_imgs_tc=Variable(torch.Tensor(test_imgs_np).cuda())
        # @ Make predictions
        prediction=model_api_instance.gen_net(test_imgs_tc)
        # print("prediction",prediction)
        # tensor([[0.3921, 0.6079],
        #         [0.5055, 0.4945],
        #         [0.2936, 0.7064],
        #         [0.5176, 0.4824],
        #         [0.6384, 0.3616],
        #         [0.3644, 0.6356],
        #         [0.6779, 0.3221],
        #         [0.7902, 0.2098],
        #         [0.1686, 0.8314],
        #         [0.6742, 0.3258],
        #         [0.6005, 0.3995]], device='cuda:0')
        # @ Make predicted labels
        arg_max_vals=prediction.argmax(1).cpu().numpy()
        # print("arg_max_vals",arg_max_vals)
        # [1 0 1 0 0 1 0 0 1 0 0]
        # print("lbls",lbls)
        # [1 0 0 0 0 1 0 0 1 0 0]
        # @ Calculate number of correct cases
        num_correct_case=np.sum(arg_max_vals==lbls)
        entire_correct_cases+=num_correct_case
      # @ Calculate true positive
      acc_true_positive=entire_correct_cases/num_imgs_test
      print("acc_true_positive",acc_true_positive)
      afaf
