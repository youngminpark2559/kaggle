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
def get_file_list(path):
    file_list=glob.glob(path)
    file_list=natsort.natsorted(file_list,reverse=False)
    return file_list

def return_path_list_from_txt(txt_file):
    txt_file=open(txt_file, "r")
    read_lines=txt_file.readlines()
    num_file=int(len(read_lines))
    txt_file.close()
    return read_lines,num_file

def return_path_list_from_txt_on_EC2(txt_file,bucket):
    import boto3
    temp_list=[]
    len_txt_file=len(txt_file)
    for _ in range(len_txt_file):
        path_of_one_img=bucket.download_file(path_of_one_img)
        temp_list.append(path_of_one_img)
    return temp_list,len_txt_file

def generate_dense_dataset(dense_d_bs_pa,img_h,img_w):
    dense_ori_ph=[]
    dense_ref_ph=[]

    for one_img in range(len(dense_d_bs_pa)):
        # c o_dense_ori_img_p: one dense train image path
        o_dense_ori_img_p=dense_d_bs_pa[one_img][0].strip()
        # print("o_dense_ori_img_p",o_dense_ori_img_p)
        # c o_dense_r_gt_img_p: one dense gt image path
        o_dense_r_gt_img_p=dense_d_bs_pa[one_img][1].strip()
        # print("o_dense_r_gt_img_p",o_dense_r_gt_img_p)

        # --------------------------------------------------------------------------------
        # c ol_dense_tr_i: one loaded dense train image
        ol_dense_tr_i=utils_image.load_img(o_dense_ori_img_p)/255.0
        # ol_dense_tr_i=(ol_dense_tr_i-np.mean(ol_dense_tr_i))/(np.std(ol_dense_tr_i)+0.000001)
        # ol_dense_tr_i=(ol_dense_tr_i-np.min(ol_dense_tr_i))/np.ptp(ol_dense_tr_i)
        # ol_dense_tr_i[ol_dense_tr_i==0.0]=0.0001
        # print("ol_dense_tr_i",ol_dense_tr_i)
        # plt.imshow(ol_dense_tr_i)
        # plt.show()

        ol_dense_R_gt_i=utils_image.load_img(o_dense_r_gt_img_p)/255.0
        # c ol_iiw_tr_i: one loaded iiw train image
        
        # --------------------------------------------------------------------------------
        # c ol_r_dense_tr_i: one loaded resized dense train image
        ol_r_dense_tr_i=utils_image.resize_img(ol_dense_tr_i,img_h,img_w)
        ol_r_dense_R_gt_i=utils_image.resize_img(ol_dense_R_gt_i,img_h,img_w)

        dense_ori_ph.append(ol_r_dense_tr_i)
        dense_ref_ph.append(ol_r_dense_R_gt_i)

    dense_ori_imgs=np.stack(dense_ori_ph,axis=0).transpose(0,3,1,2)
    dense_ref_imgs=np.stack(dense_ref_ph,axis=0).transpose(0,3,1,2)

    return dense_ori_imgs,dense_ref_imgs

def generate_sparse_dataset(iiw_d_bs_pa,img_h,img_w):
    iiw_tr_ph=[]
    iiw_ori_ph=[]
    iiw_json_ph=[]
    applied_DAs=[]
    iiw_img_after_DA_before_rs_ph=[]
    for one_img in range(len(iiw_d_bs_pa)):
        # c one_iiw_tr_img_p: one iiw train image path
        one_iiw_tr_img_p=iiw_d_bs_pa[one_img][0].strip()
        # print("one_iiw_tr_img_p",one_iiw_tr_img_p)

        # c one_iiw_json_gt_p: one iiw json gt path
        one_iiw_json_gt_p=iiw_d_bs_pa[one_img][1].strip()
        # print("one_iiw_json_gt_p",one_iiw_json_gt_p)

        # --------------------------------------------------------------------------------
        ol_iiw_tr_i=utils_image.load_img(one_iiw_tr_img_p)/255.0
        # ol_iiw_tr_i=np.where(ol_iiw_tr_i==0.0,0.0001,ol_dense_tr_i)
        # ol_iiw_tr_i[ol_iiw_tr_i==0.0]=0.0001

        # --------------------------------------------------------------------------------
        # c kind_of_DA: you create list which contains kind of data augmentation
        kind_of_DA=["no_DA","ud","lr","p3","p6","p9","n3","n6","n9"]
        # c chosen_DA: you get chosen kind of data augmentation
        chosen_DA=np.random.choice(kind_of_DA,1,replace=False)[0]
        # print("chosen_DA",chosen_DA)
        # chosen_DA n9

        if chosen_DA=="ud":
            iiw_img_after_DA=np.flipud(ol_iiw_tr_i)
        elif chosen_DA=="lr":
            iiw_img_after_DA=np.fliplr(ol_iiw_tr_i)
        elif chosen_DA=="p3":
            iiw_img_after_DA=scipy.ndimage.interpolation.rotate(ol_iiw_tr_i,angle=3,reshape=True,mode="reflect")
        elif chosen_DA=="p6":
            iiw_img_after_DA=scipy.ndimage.interpolation.rotate(ol_iiw_tr_i,angle=6,reshape=True,mode="reflect")
        elif chosen_DA=="p9":
            iiw_img_after_DA=scipy.ndimage.interpolation.rotate(ol_iiw_tr_i,angle=9,reshape=True,mode="reflect")
        elif chosen_DA=="n3":
            iiw_img_after_DA=scipy.ndimage.interpolation.rotate(ol_iiw_tr_i,angle=-3,reshape=True,mode="reflect")
        elif chosen_DA=="n6":
            iiw_img_after_DA=scipy.ndimage.interpolation.rotate(ol_iiw_tr_i,angle=-6,reshape=True,mode="reflect")
        elif chosen_DA=="n9":
            iiw_img_after_DA=scipy.ndimage.interpolation.rotate(ol_iiw_tr_i,angle=-9,reshape=True,mode="reflect")
        else:
            iiw_img_after_DA=ol_iiw_tr_i
            
        # plt.imshow(iiw_img_after_DA)
        # plt.show()
        # afaf

        iiw_img_after_DA=cv2.normalize(
            iiw_img_after_DA,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        iiw_img_after_DA[iiw_img_after_DA<1e-6]=1e-6

        iiw_img_after_DA_before_rs=iiw_img_after_DA

        # --------------------------------------------------------------------------------
        iiw_img_after_DA_after_rs=utils_image.resize_img(iiw_img_after_DA,img_h,img_w)
        # plt.imshow(iiw_img_after_DA_after_rs)
        # plt.show()
        # afaf

        # --------------------------------------------------------------------------------
        # iiw_ori_ph: original size
        iiw_ori_ph.append(ol_iiw_tr_i)
        # iiw_tr_ph: resized size
        iiw_tr_ph.append(iiw_img_after_DA_after_rs)
        iiw_json_ph.append(one_iiw_json_gt_p)
        applied_DAs.append(chosen_DA)
        iiw_img_after_DA_before_rs_ph.append(iiw_img_after_DA_before_rs)

    iiw_tr_imgs=np.stack(iiw_tr_ph,axis=0).transpose(0,3,1,2)

    return iiw_tr_imgs,iiw_ori_ph,iiw_json_ph,applied_DAs,iiw_img_after_DA_before_rs_ph

def generate_cgmit_dataset(cgmit_d_bs_pa,img_h,img_w):
    cgmit_tr_3c_img=[]
    cgmit_gt_R_3c_img=[]
    cgmit_gt_S_1c_img=[]
    cgmit_mask_3c_img=[]

    for one_img in range(len(cgmit_d_bs_pa)):
        # c one_iiw_tr_img_p: one iiw train image path
        one_cgmit_tr_img_p=cgmit_d_bs_pa[one_img][0].strip()
        # print("one_cgmit_tr_img_p",one_cgmit_tr_img_p)
        
        # c one_cgmit_r_gt_p: one iiw json gt path
        one_cgmit_r_gt_p=cgmit_d_bs_pa[one_img][1].strip()
        # print("one_cgmit_r_gt_p",one_cgmit_r_gt_p)

        # --------------------------------------------------------------------------------
        ol_cgmit_tr_i=utils_image.load_img(one_cgmit_tr_img_p)/255.0
        # ol_iiw_tr_i=np.where(ol_iiw_tr_i==0.0,0.0001,ol_dense_tr_i)
        # ol_iiw_tr_i[ol_iiw_tr_i==0.0]=0.0001

        ol_cgmit_r_gt_i=utils_image.load_img(one_cgmit_r_gt_p)/255.0
        
        # --------------------------------------------------------------------------------
        ol_cgmit_tr_i=utils_image.resize_img(ol_cgmit_tr_i,img_h,img_w)

        ol_cgmit_r_gt_i=utils_image.resize_img(ol_cgmit_r_gt_i,img_h,img_w)

        # --------------------------------------------------------------------------------
        # c srgb_img: actually rgb original image than srgb original image
        srgb_img=ol_cgmit_tr_i

        gt_R=ol_cgmit_r_gt_i

        mask=np.ones((srgb_img.shape[0],srgb_img.shape[1]))
        # print("mask",mask.shape)
        # mask (341, 512)

        # c gt_R_gray: mean of R gt image
        gt_R_gray=np.mean(gt_R,2)
        mask[gt_R_gray<1e-6]=0
        # mean of original image
        mask[np.mean(srgb_img,2)<1e-6]=0

        # plt.imshow(mask,cmap="gray")
        # plt.show()
        mask=skimage.morphology.binary_erosion(mask,square(11))
        mask=np.expand_dims(mask,axis=2)
        mask=np.repeat(mask,3,axis=2)
        gt_R[gt_R<1e-6]=1e-6

        rgb_img=srgb_img
        gt_S=rgb_img/gt_R
        # plt.imshow(gt_S[:,:,0],cmap='gray')
        # plt.show()

        # search_name=path[:-4]+".rgbe"
        # irridiance=self.stat_dict[search_name]

        # if irridiance<0.25:
        #     srgb_img=denoise_tv_chambolle(srgb_img,weight=0.05,multichannel=True)
        #     gt_S=denoise_tv_chambolle(gt_S,weight=0.1,multichannel=True)

        mask[gt_S>10]=0
        gt_S[gt_S>20]=20
        mask[gt_S<1e-4]=0
        gt_S[gt_S<1e-4]=1e-4

        if np.sum(mask)<10:
            max_S=1.0
        else:
            max_S=np.percentile(gt_S[mask>0.5],90)

        gt_S=gt_S/max_S

        gt_S=np.mean(gt_S,2)
        gt_S=np.expand_dims(gt_S,axis=2)

        #gt_R=np.mean(gt_R,2)
        gt_R=np.expand_dims(gt_R,axis=2)

        tr_3c_img=srgb_img.squeeze()
        gt_R_3c_img=gt_R.squeeze()
        gt_S_1c_img=gt_S.squeeze()[:,:,np.newaxis]
        mask_3c_img=mask.squeeze()
        # print("tr_3c_img",tr_3c_img.shape)
        # print("gt_R_3c_img",gt_R_3c_img.shape)
        # print("gt_S_1c_img",gt_S_1c_img.shape)
        # print("mask_3c_img",mask_3c_img.shape)
        # tr_3c_img (1024, 1024, 3)
        # gt_R_3c_img (1024, 1024, 3)
        # gt_S_1c_img (1024, 1024, 1)
        # mask_3c_img (1024, 1024, 3)

        # --------------------------------------------------------------------------------
        cgmit_tr_3c_img.append(tr_3c_img)
        cgmit_gt_R_3c_img.append(gt_R_3c_img)
        cgmit_gt_S_1c_img.append(gt_S_1c_img)
        cgmit_mask_3c_img.append(mask_3c_img.astype("float16"))

    cgmit_tr_3c_imgs=np.stack(cgmit_tr_3c_img,axis=0).transpose(0,3,1,2)
    cgmit_gt_R_3c_imgs=np.stack(cgmit_gt_R_3c_img,axis=0).transpose(0,3,1,2)
    cgmit_gt_S_1c_imgs=np.stack(cgmit_gt_S_1c_img,axis=0).transpose(0,3,1,2)
    cgmit_mask_3c_imgs=np.stack(cgmit_mask_3c_img,axis=0).transpose(0,3,1,2)
    # print("cgmit_tr_3c_imgs",cgmit_tr_3c_imgs.shape)
    # print("cgmit_gt_R_3c_imgs",cgmit_gt_R_3c_imgs.shape)
    # print("cgmit_gt_S_1c_imgs",cgmit_gt_S_1c_imgs.shape)
    # print("cgmit_mask_3c_imgs",cgmit_mask_3c_imgs.shape)
    # cgmit_tr_3c_imgs (4, 3, 1024, 1024)
    # cgmit_gt_R_3c_imgs (4, 3, 1024, 1024)
    # cgmit_gt_S_1c_imgs (4, 1, 1024, 1024)
    # cgmit_mask_3c_imgs (4, 3, 1024, 1024)

    return cgmit_tr_3c_imgs,cgmit_gt_R_3c_imgs,cgmit_gt_S_1c_imgs,cgmit_mask_3c_imgs

def processing_for_cgmit_dataset(cgmit_ori_li,cgmit_rgt_li):
    cgmit_tr_3c_img=[]
    cgmit_gt_R_3c_img=[]
    cgmit_gt_S_1c_img=[]
    cgmit_mask_3c_img=[]

    # print("cgmit_ori_li",cgmit_ori_li.shape)
    # print("cgmit_rgt_li",cgmit_rgt_li.shape)
    # cgmit_ori_li (2, 3, 512, 512)
    # cgmit_rgt_li (2, 3, 512, 512)

    for one_img in range(cgmit_ori_li.shape[0]):
        # --------------------------------------------------------------------------------
        # c srgb_img: actually rgb original image than srgb original image
        rgb_img=cgmit_ori_li[one_img,:,:,:].transpose(1,2,0)
        # print("rgb_img",rgb_img.shape)
        # rgb_img (512, 512, 3)
        srgb_img=utils_image.rgb_to_srgb(rgb_img)

        gt_R=cgmit_rgt_li[one_img,:,:,:].transpose(1,2,0)
        # print("gt_R",gt_R.shape)
        # gt_R (512, 512, 3)

        mask=np.ones((srgb_img.shape[0],srgb_img.shape[1]))
        # print("mask",mask.shape)
        # mask (341, 512)
        # mask[0,0]=0.0

        # c gt_R_gray: mean of R gt image
        gt_R_gray=np.mean(gt_R,2)
        mask[gt_R_gray<1e-4]=0.0

        # mean of original image
        srgb_img_gray=np.mean(srgb_img,2)
        mask[srgb_img_gray<1e-4]=0.0
        
        # plt.imshow(mask,cmap="gray")
        # plt.show()
        mask=skimage.morphology.binary_erosion(mask,square(10))
        mask=np.expand_dims(mask,axis=2)
        mask=np.repeat(mask,3,axis=2)

        # print("np.min(srgb_img)",np.min(srgb_img))
        # print("np.max(srgb_img)",np.max(srgb_img))
        # print("np.min(gt_R)",np.min(gt_R))
        # print("np.max(gt_R)",np.max(gt_R))
        # np.min(srgb_img) 0.0
        # np.max(srgb_img) 1.0
        # np.min(gt_R) 1e-06
        # np.max(gt_R) 0.8563725490196078
        gt_R[gt_R<1e-6]=1e-6
        # print("np.min(gt_R_temp)",np.min(gt_R_temp))
        # print("np.max(gt_R_temp)",np.max(gt_R_temp))
        # print("np.min(rgb_img)",np.min(rgb_img))
        # print("np.max(rgb_img)",np.max(rgb_img))
        # np.min(gt_R_temp) 1e-06
        # np.max(gt_R_temp) 0.9764705882352941
        # np.min(rgb_img) 0.0
        # np.max(rgb_img) 0.7137254901960784
        
        gt_S=rgb_img/gt_R
        gt_S=np.clip(gt_S,0.0,1.3)
        # min_v=np.min(gt_S)
        # range_v=np.max(gt_S)-min_v
        # if range_v>0:
        #     gt_S=(gt_S-min_v)/range_v
        # else:
        #     gt_S=torch.zeros(gt_S.size())
        # gt_S=(gt_S-np.mean(gt_S))/np.std(gt_S)
        # gt_S=(gt_S-np.min(gt_S))/np.ptp(gt_S)
        gt_S=cv2.normalize(gt_S,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)[:,:,0]
        # plt.subplot(2,2,1)
        # plt.imshow(gt_R)
        # plt.subplot(2,2,2)
        # plt.imshow(rgb_img)
        # plt.subplot(2,2,3)
        # plt.imshow(gt_S[:,:,0],cmap='gray')
        # plt.subplot(2,2,4)
        # plt.imshow(mask[:,:,0],cmap='gray')
        # plt.show()

        # search_name=path[:-4]+".rgbe"
        # irridiance=self.stat_dict[search_name]

        # if irridiance<0.25:
        #     srgb_img=denoise_tv_chambolle(srgb_img,weight=0.05,multichannel=True)
        #     gt_S=denoise_tv_chambolle(gt_S,weight=0.1,multichannel=True)

        # mask[gt_S>10]=0
        # gt_S[gt_S>20]=20
        # mask[gt_S<1e-4]=0
        # gt_S[gt_S<1e-4]=1e-4

        # if np.sum(mask)<10:
        #     max_S=1.0
        # else:
        #     max_S=np.percentile(gt_S[mask>0.5],90)

        # gt_S=gt_S/max_S

        # gt_S=np.mean(gt_S,2)
        # gt_S=np.expand_dims(gt_S,axis=2)

        # # gt_R=np.mean(gt_R,2)
        # # gt_R=np.expand_dims(gt_R,axis=2)

        # plt.subplot(2,2,1)
        # plt.imshow(gt_R)
        # plt.subplot(2,2,2)
        # plt.imshow(rgb_img)
        # plt.subplot(2,2,3)
        # plt.imshow(gt_S,cmap='gray')
        # plt.subplot(2,2,4)
        # plt.imshow(mask[:,:,0],cmap='gray')
        # plt.show()
        # scipy.misc.imsave("./mask.png",mask[:,:,0].astype("float"))


        tr_3c_img=rgb_img.squeeze()
        gt_R_3c_img=gt_R.squeeze()
        gt_S_1c_img=gt_S.squeeze()[:,:,np.newaxis]
        mask_3c_img=mask.squeeze()
        # print("tr_3c_img",tr_3c_img.shape)
        # print("gt_R_3c_img",gt_R_3c_img.shape)
        # print("gt_S_1c_img",gt_S_1c_img.shape)
        # print("mask_3c_img",mask_3c_img.shape)
        # tr_3c_img (512, 512, 3)
        # gt_R_3c_img (512, 512, 3)
        # gt_S_1c_img (512, 512, 1, 3)
        # mask_3c_img (512, 512, 3)

        # --------------------------------------------------------------------------------
        cgmit_tr_3c_img.append(tr_3c_img)
        cgmit_gt_R_3c_img.append(gt_R_3c_img)
        cgmit_gt_S_1c_img.append(gt_S_1c_img)
        cgmit_mask_3c_img.append(mask_3c_img.astype("float"))

    cgmit_tr_3c_imgs=np.stack(cgmit_tr_3c_img,axis=0).transpose(0,3,1,2)
    cgmit_gt_R_3c_imgs=np.stack(cgmit_gt_R_3c_img,axis=0).transpose(0,3,1,2)
    cgmit_gt_S_1c_imgs=np.stack(cgmit_gt_S_1c_img,axis=0).transpose(0,3,1,2)
    cgmit_mask_3c_imgs=np.stack(cgmit_mask_3c_img,axis=0).transpose(0,3,1,2)
    # print("cgmit_tr_3c_imgs",cgmit_tr_3c_imgs.shape)
    # print("cgmit_gt_R_3c_imgs",cgmit_gt_R_3c_imgs.shape)
    # print("cgmit_gt_S_1c_imgs",cgmit_gt_S_1c_imgs.shape)
    # print("cgmit_mask_3c_imgs",cgmit_mask_3c_imgs.shape)
    # cgmit_tr_3c_imgs (4, 3, 1024, 1024)
    # cgmit_gt_R_3c_imgs (4, 3, 1024, 1024)
    # cgmit_gt_S_1c_imgs (4, 1, 1024, 1024)
    # cgmit_mask_3c_imgs (4, 3, 1024, 1024)

    return cgmit_tr_3c_imgs,cgmit_gt_R_3c_imgs,cgmit_gt_S_1c_imgs,cgmit_mask_3c_imgs

def chunks(l,n):
    # For item i in range that is length of l,
    for i in range(0,len(l),n):
        # Create index range for l of n items:
        yield l[i:i+n]

def divisorGenerator(n):
    large_divisors=[]
    for i in range(1,int(math.sqrt(n)+1)):
        if n%i==0:
            yield i
            if i*i!=n:
                large_divisors.append(n/i)
    for divisor in reversed(large_divisors):
        yield int(divisor)
# list(divisorGenerator(1024))
