# conda activate py36gputorch041
# cd /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/utils/
# rm e.l && python utils_image.py 2>&1 | tee -a e.l && code e.l

# @ Basic modules
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
import skimage
from skimage.morphology import square
from skimage.restoration import denoise_tv_chambolle
import timeit
import sys,os
import glob
import natsort 
from itertools import zip_longest # for Python 3.x
import math
# @ PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
# @ src/networks
from src.networks import networks as networks
# @ src/utils
from src.utils import utils_image as utils_image

class CAM():
  def __init__(self,model):
    # c gradient: list for gradient placeholder
    self.gradient=[]
    # c h: save_gradient() is called when model.module.layer[-1] is called in backward step
    # self.h = model.module.layer[-1].register_backward_hook(self.save_gradient)
    # c h: save_gradient() is called when model.end_conv is called in backward step
    # self.h = model.end_conv.register_backward_hook(self.save_gradient)
    # c h: save_gradient() is called when model.end_conv is called in forward step
    self.h = model.end_conv.register_forward_hook(self.save_gradient)

  def save_gradient(self,*args):
    """
    Act
      * This function is automatically called to save gradient values
      when last layer is being processed in backward step
    Params
      * 
    Return
      * 
    """
    grad_input=args[1]
    grad_output=args[2]
    # print("grad_input",grad_input.shape)
    # print("grad_output",grad_output.shape)
    # c gradient: append grad_output into gradient list
    self.gradient.append(grad_output[0])
      
  def get_gradient(self,idx):
    """
    Act
      * This function returns gradient value of specific index
    Params
      * idx
    Return
      * 
    """
    # print(np.array(self.gradient).shape)
    # (9,)
    # print(self.gradient)
    return self.gradient[idx]
  
  def remove_hook(self):
    """
    Act
      * This function removes registered hook function 
    Params
      * 
    Return
      * 
    """
    self.h.remove()
          
  def normalize_cam(self,x):
    """
    Act
      * This function normalizes CAM
    Params
      * x
    Return
      * 
    """
    min_of_x=torch.min(x)
    max_of_x=torch.max(x)
    under_term=(min_of_x-max_of_x+1e-8)-1
    upper_term=2*(x-min_of_x)
    x=upper_term/under_term
    x[x<torch.max(x)]=-1
    return x
  
  def visualize(self,cam_img,img_var):
    """
    Act
      * 
    Params
      * cam_img
      * img_var
    Return
      * 
    """
    # c cam_img: [7, 7]
    # c img_var: [1, 28, 28]
    # te_img=cam_img.detach().cpu().numpy()
    # plt.imshow(te_img,cmap='gray')
    # plt.show()
    # c cam_img: resized cam_img
    cam_img=resize(cam_img.cpu().data.numpy(),output_shape=(28,28))
    # plt.imshow(cam_img,cmap='gray')
    # plt.show()
    x=img_var[0,:,:].cpu().data.numpy()
    # plt.subplot(1,3,1)
    # plt.imshow(cam_img)
    # plt.subplot(1,3,2)
    # plt.imshow(x,cmap="gray")
    # plt.subplot(1,3,3)
    # plt.imshow(x+cam_img)
    # plt.show()
  
  def get_cam(self,idx):
    # print("self.get_gradient",self.get_gradient)
    # <bound method CAM.get_gradient of <__main__.CAM object at 0x7f11bbcabeb8>>
    grad=self.get_gradient(idx)
    # print("grad",grad)
    # print("grad",grad.shape)
    # [[[[-1.3758e-02, -2.0786e-02,  2.1556e-02,  ...,  1.1455e-02,
    #      8.9795e-03, -4.4681e-03],
    # [2, 1, 256, 256]
    # [1, 1024, 1024]
    print("len(grad.shape)",len(grad.shape))
    # len(grad.shape) 3
    if len(grad.shape)==3:
      grad=torch.unsqueeze(grad, 1)
    # print("grad",grad.shape)
    # torch.Size([1, 1, 1024, 1024])
    alpha=torch.sum(grad,dim=3,keepdim=True)
    # print("alpha",alpha)
    # print("alpha",alpha.shape)
    # [[[[-0.0067],
    #    [ 0.0315],
    #    [-0.0135],
    # [2, 1, 256, 1]
    alpha=torch.sum(alpha,dim=2,keepdim=True)
    # print("alpha",alpha)
    # print("alpha",alpha.shape)
    # [[[[-0.0561]],
    #   [[-0.0587]],
    #   [[ 0.0461]],
    # [2, 1, 1, 1]
    # cam = alpha[j]*grad[j]
    cam=alpha[idx]*grad[idx]
    # print("cam",cam)
    # [[[ 7.5869e-10,  1.2213e-09,  1.6948e-09,  ...,  2.7088e-09, 2.4616e-09,  2.2293e-09],
    #   [ 1.1486e-09,  1.6282e-09,  2.1207e-09,  ...,  3.0267e-09, 2.5283e-09,  2.0287e-09],
    # print("cam",cam.shape)
    # [1, 256, 256]
    cam=torch.sum(cam,dim=0)
    # print("cam",cam)
    # [[ 7.5869e-10,  1.2213e-09,  1.6948e-09,  ...,  2.7088e-09, 2.4616e-09,  2.2293e-09],
    #  [ 1.1486e-09,  1.6282e-09,  2.1207e-09,  ...,  3.0267e-09, 2.5283e-09,  2.0287e-09],
    # print("cam",cam.shape)
    # [256, 256]
    cam=self.normalize_cam(cam)
    # print("cam",cam.shape)
    # [[-1., -1., -1.,  ..., -1., -1., -1.],
    #  [-1., -1., -1.,  ..., -1., -1., -1.],
    # print("cam",cam)
    # [256, 256]
    te_img=cam.detach().cpu().numpy()
    plt.imshow(te_img,cmap='gray')
    plt.show()
    # @ Remove hook
    self.remove_hook()
    return cam
