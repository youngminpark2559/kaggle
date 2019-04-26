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
import skimage
from skimage.morphology import square
from skimage.restoration import denoise_tv_chambolle
import timeit
import sys,os
import glob
import natsort 
from itertools import zip_longest # for Python 3.x
import math

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.optim import lr_scheduler
# from torch.autograd import Variable
# from torch.utils.data import Dataset, DataLoader
# import torchvision
# import torchvision.transforms as transforms
# from torchvision import datasets, models, transforms

# ================================================================================
# network_dir="./networks"
# sys.path.insert(0,network_dir)
# import networks as networks
from src.networks import networks as networks

# utils_dir="./utils"
# sys.path.insert(0,utils_dir)
# import utils_image as utils_image
from src.utils import utils_image as utils_image

# ================================================================================
def printnorm(self, input, output):
    """
    Act
      * Hook function which is used in forward step
    
    Params
      * input (tuple packed inputs)
      * output (Tensor)
      output.data is Tensor we are interested
    
    Return
      * 
    """
    print("Start: forward hook function on conv2")
    print('Let\'s see how ' + self.__class__.__name__ + ' forward step works')
    # print('data type of input which is passed into conv2: ', type(input))
    # tuple
    # print('shape of input which is passed into conv2: ', np.array(input).shape)
    # (1,)
    # print('shape of input data which is passed into conv2: ', input[0].shape)
    # [64, 20, 12, 12]
    print('shape of output data which is output from conv2: ', output[0].shape)
    # [50, 8, 8]
    print('norm of output.data :', output.data.norm())
    print("End: forward hook function on conv2\n\n")

def printgradnorm(self, grad_input, grad_output):
    """
    Act
      * Hook function which is used in backward step
    
    Params
      * grad_input
      * grad_output
    
    Return
      * 
    """
    print("Start: backward hook function on conv2")
    print('Let\'s see how ' + self.__class__.__name__ + ' backward step works')
    print('Inside class: ' + self.__class__.__name__)
    print('grad_input which is passed into conv2',np.array(grad_input).shape)
    # (3,)
    print('grad_input[0] which is passed into conv2',grad_input[0].shape)
    # [64, 20, 12, 12]
    # print("grad_input[0].shape[0]",grad_input[0].shape[0])
    # 64
    
    # grad_input_copy=grad_input[0].detach().cpu().numpy().copy()
    # # print('grad_input_copy',grad_input_copy.shape)
    # # grad_input_copy (64, 20, 12, 12)
    # for i in range(grad_input_copy.shape[1]):
    #     grad_input_img=grad_input_copy[0,i,:,:]
    #     # print("grad_input_img",grad_input_img.shape)
    #     # (12, 12)
    #     grad_input_img_normalized=(grad_input_img-np.min(grad_input_img))/np.ptp(grad_input_img)
    #     # print("grad_input_img",grad_input_img)
    #     scipy.misc.imsave('grad_input_img_normalized_'+str(i)+'.png', grad_input_img_normalized)
    #     # plt.imshow(grad_input_img_normalized,cmap='gray')
    #     # plt.show()

    grad_output_copy=grad_output[0].detach().cpu().numpy().copy()
    # print('grad_output_copy',grad_output_copy.shape)
    # grad_output_copy (64, 20, 12, 12)
    for i in range(grad_output_copy.shape[1]):
        grad_output_img=grad_output_copy[0,i,:,:]
        # print("grad_output_img",grad_output_img.shape)
        # (12, 12)
        grad_output_img_normalized=(grad_output_img-np.min(grad_output_img))/np.ptp(grad_output_img)
        # print("grad_output_img",grad_output_img)
        scipy.misc.imsave('grad_output_img_normalized_'+str(i)+'.png', grad_output_img_normalized)
        # plt.imshow(grad_output_img_normalized,cmap='gray')
        # plt.show()
    afaf
    
    print('grad_input[1] which is passed into conv2',grad_input[1].shape)
    # [50, 20, 5, 5]
    print('grad_input[2] which is passed into conv2',grad_input[2].shape)
    # [50]
    print('grad_output which is passed into conv2',np.array(grad_output).shape)
    # (1,)
    print('grad_output[0] which is passed into conv2',grad_output[0].shape)
    # [64, 50, 8, 8]
    print('norm of grad_input[0]:', grad_input[0].norm())
    # tensor(0.0135, device='cuda:0')
    print("End: backward hook function on conv2\n\n")
