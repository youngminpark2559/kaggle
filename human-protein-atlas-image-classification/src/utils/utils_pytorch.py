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

import torch
from torch.autograd import Variable

# ================================================================================
from src.networks import networks as networks

from src.utils import utils_image as utils_image

# ================================================================================
def get_Variable(ori_imgs):
  dense_O_img_tc=Variable(torch.Tensor(ori_imgs).cuda())
  return dense_O_img_tc
