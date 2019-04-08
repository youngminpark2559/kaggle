# conda activate py36gputorch041
# cd /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/utils/
# rm e.l && python utils_image.py 2>&1 | tee -a e.l && code e.l

from PIL import Image
import PIL.ImageOps
import scipy.misc
import cv2
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import image
import timeit
import sys,os
import glob
import natsort 
import traceback

import torch

# ================================================================================
from src.networks import networks as networks

# ================================================================================
if torch.cuda.is_available():
  device="cuda:0"
else:
  device="cpu"

# device=torch.device(device)
device=torch.device("cuda:0")

# ================================================================================
# Horovod: initialize library.
# hvd.init()
# torch.manual_seed(args.seed)

# if args.use_multi_gpu=="True":
#   # Horovod: pin GPU to local rank.
#   torch.cuda.set_device(hvd.local_rank())
#   torch.cuda.manual_seed(args.seed)

# ================================================================================
gpu_ids=0
use_dropout=False
ngf=64
input_nc=3
output_nc=3
LR=0.01
WD=0.1

# ================================================================================
def get_file_list(path):
  file_list=glob.glob(path)
  file_list=natsort.natsorted(file_list,reverse=False)
  return file_list

def print_network(net,struct=False):
  """
  Args
    net: created network
    struct (False): do you want to see structure of entire network?
  Print
    Structure of entire network
    Total number of parameters of network
  """
  if struct==True:
    print(net)
  num_params=0
  for param in net.parameters():
    num_params+=param.numel()
  return num_params

def net_generator(args,num_entire_imgs):
  """
  Act
    * 
  
  Params
    * args
    You can manually create args
    by using utils_create_argument.return_argument()

    * num_entire_imgs
    Example is 8
  
  Return
    * 
  """
  lr=0.0001

  if args.train_method=="train_by_transfer_learning_using_resnet":  
    # gen_net=networks.Pretrained_ResNet152().cuda()
    gen_net=networks.Pretrained_ResNet50().cuda()
    # gen_net=networks.Pretrained_VGG16().cuda()
    # gen_net=networks.Custom_Net().cuda()
    

    # ================================================================================
    if args.use_multi_gpu=="True":
      num_gpu=torch.cuda.device_count()
      print("num_gpu",num_gpu)
      # DEVICE_IDS=list(range(num_gpu))
      # DEVICE_IDS=[0,1,2,3,4,5,6,7]
      # gen_encoder=nn.DataParallel(gen_encoder,device_ids=DEVICE_IDS)
      gen_net=nn.DataParallel(gen_net)
    else: # args.use_multi_gpu=="False":
      pass

    # ================================================================================
    # Optimizer and scheduler

    optimizer=torch.optim.Adam(gen_net.parameters(),lr=lr)

    # ================================================================================
    # Load model

    if args.use_saved_model_for_continuous_train=="True":
      checkpoint_gener_direct_rgb=torch.load(
        args.model_save_dir+args.model_file_name_when_saving_and_loading_model)

      start_epoch=checkpoint_gener_direct_rgb['epoch']

      gen_net.load_state_dict(checkpoint_gener_direct_rgb['state_dict'])

      optimizer.load_state_dict(checkpoint_gener_direct_rgb['optimizer'])
    
    # ================================================================================
    # Print network infomation

    gen_net_param=print_network(gen_net)

    # gen_net_param=print_network(gen_net_test)
    print("gen_net:",gen_net_param)

    return gen_net,optimizer

  # ================================================================================
  else: # use_integrated_decoders=="False":
    pass

# ================================================================================
def save_checkpoint(state,filename):
    torch.save(state,filename)

# ================================================================================
def denorm(x):
    out=(x+1)/2
    return out.clamp(0,1)

# ================================================================================
def remove_existing_gradients_before_starting_new_training(
  model_list,args):

  if args.use_integrated_decoders=="True":
    if args.scheduler=="cyclic":
      model_list["gen_net"].zero_grad()
      # scheduler_encoder.batch_step()
    elif args.scheduler=="cosine":
      model_list["optimizer"].zero_grad()
      # scheduler_encoder.step()
    else:
      model_list["gen_net"].zero_grad()
  else:
    pass

# ================================================================================
def save_trained_model_after_training_epoch(one_ep,model_list,args):
  
  if args.use_integrated_decoders=="True":

    gen_net=model_list["gen_net"]
    optimizer=model_list["optimizer"]

    # ================================================================================
    if not os.path.exists(args.model_save_dir):
      os.makedirs(args.model_save_dir)

    # ================================================================================
    model_name=args.model_file_name_when_saving_and_loading_model.split(".")[0]
    net_path=args.model_save_dir+model_name+"_"+str(one_ep)+".pth"

    # ================================================================================
    save_checkpoint(
      {'epoch':one_ep+1,
       'state_dict':gen_net.state_dict(),
       'optimizer':optimizer.state_dict()},
      net_path)

  else:
    pass

# ================================================================================
def empty_cache_of_gpu_after_training_batch():
  num_gpus=torch.cuda.device_count()
  for gpu_id in range(num_gpus):
    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()
    # print("empty cache gpu in dense")
