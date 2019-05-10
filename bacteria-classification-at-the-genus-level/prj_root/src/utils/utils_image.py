# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/prj_root/src/utils && \
# rm e.l && python utils_image.py 2>&1 | tee -a e.l && code e.l

import matplotlib as mpl
from PIL import Image
import PIL.ImageOps
import cv2
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import image
import timeit
import sys
import scipy.misc
import warnings
from skimage.transform import resize
from scipy.ndimage import convolve
from skimage.restoration import denoise_tv_chambolle

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
from torchvision import transforms
from torch.autograd import Variable

# ================================================================================
# currentdir = "/mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/utils"
# network_dir="/mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/networks"
# sys.path.insert(0,network_dir)

currentdir = "/mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/prj_root"
sys.path.insert(0,currentdir)

# import networks as networks

from src.utils import utils_common as utils_common
from src.utils import utils_image as utils_image

# ================================================================================
def i_show_plt(img,gray=False):
    if gray==False:
        plt.imshow(img)
        plt.show()
    else:
        plt.imshow(img,cmap="gray")
        plt.show()

def resize_img(img,img_h,img_w):
    # print("img",img.shape)
    # img (348, 819, 3)
    rs_img=resize(img,(img_h,img_w),mode='constant',anti_aliasing=True,anti_aliasing_sigma=None)
    # print("rs_img",rs_img.shape)
    # rs_img (512, 512, 3)

    return rs_img

def load_img(path,gray=False):
    path=path.replace("\n","")
    if gray==True:
        img=Image.open(path).convert("L")
        img=np.array(img)
    else:
        img=Image.open(path)
        img=np.array(img)
    return img

def create_guided_img(ori_img,target_img,radius=50,eps=50):
    guided_img=cv2.ximgproc.guidedFilter(
        ori_img, target_img, radius, eps)
    return guided_img

def bilateral_f(guided_img,target_img,d=-1,sigmaColor=20,sigmaSpace=20):
    """
    Dont use guided filter
    guided_img original
    target_img r
    """
    filtered = cv2.ximgproc.jointBilateralFilter(
        guided_img,target_img,d,sigmaColor,sigmaSpace)
    return filtered

def DTF(guided_img,target_img,sigmaSpatial=40,sigmaColor=40):
    """
    Dont use guided filter
    guided_img original
    target_img r
    """
    dst=cv2.ximgproc.dtFilter(
        guided_img,target_img,sigmaSpatial,sigmaColor)
    return dst

def rgb_to_chromaticity(rgb):
    """ converts rgb to chromaticity """
    # print("rgb",rgb.shape)
    # rgb (512, 512, 3)
    irg=np.zeros_like(rgb)
    s=np.sum(rgb,axis=-1)+1e-6

    irg[..., 0]=rgb[...,0]/s
    irg[..., 1]=rgb[...,1]/s
    irg[..., 2]=rgb[...,2]/s

    return irg

def rgb_to_chromaticity_torch(torch_rgb):
    """ converts rgb to chromaticity """
    # same_cuda=torch.device('cuda:'+str(torch_rgb.get_device()))

    # irg=torch.zeros(torch_rgb.shape).cuda(same_cuda)
    irg=torch.zeros(torch_rgb.shape).cuda()

    # s=torch.sum(torch_rgb,dim=1)+torch.Tensor([1e-6]).squeeze().cuda(same_cuda)
    s=torch.sum(torch_rgb,dim=1)+torch.Tensor([1e-6]).squeeze().cuda()

    irg[:,0,:,:]=torch.div(torch_rgb[:,0,:,:],s)
    irg[:,1,:,:]=torch.div(torch_rgb[:,1,:,:],s)
    irg[:,2,:,:]=torch.div(torch_rgb[:,2,:,:],s)

    return irg

def torch_rgb_to_torch_irg(torch_rgb):
    """ converts rgb to (mean of channels, red chromaticity, green chromaticity) """
    # same_cuda=torch.device('cuda:'+str(torch_rgb.get_device()))

    # irg=torch.zeros(torch_rgb.shape).cuda(same_cuda)
    irg=torch.zeros(torch_rgb.shape).cuda()
    # print("irg",irg.shape)
    # irg torch.Size([2, 3, 528, 528])

    s=torch.sum(torch_rgb,dim=1)

    # irg[:,0,:,:]=torch.div(s,torch.Tensor([3.0]).cuda(same_cuda))
    irg[:,0,:,:]=torch.div(s,torch.Tensor([3.0]).cuda())
    irg[:,1,:,:]=torch.div(torch_rgb[:,0,:,:],s)
    irg[:,2,:,:]=torch.div(torch_rgb[:,1,:,:],s)
    # print("irg",irg.shape)
    # irg torch.Size([2, 3, 528, 528])

    return irg

# def colorize(intensity,image,eps=1e-4):
#     # Mean value of color channel
#     # It becomes 1 channel grayscale image
#     norm_input=np.mean(image,axis=2)
#     shading=norm_input/intensity
#     reflectance=image/np.maximum(shading,eps)[:,:,np.newaxis]
#     return reflectance,shading

def colorize(intensity,image,eps=1e-4):
    intensity[np.abs(intensity)<1e-4]=1e-4
    # Mean value of color channel
    # It becomes 1 channel grayscale image
    norm_input=np.mean(image,axis=2)
    shading=norm_input/intensity
    reflectance=image/np.maximum(shading,eps)[:,:,np.newaxis]
    return reflectance,shading

# def colorize_tc(intensity,image,eps=1e-4):
#     # batch,channel,H,W
#     intensity=torch.where(
#         torch.abs(intensity)<1e-4,
#         torch.Tensor([eps]).squeeze().cuda(),intensity)
#     norm_input=torch.mean(image,dim=1,keepdim=True)
#     # print("norm_input",norm_input.shape)
#     # norm_input torch.Size([10, 1, 512, 512])
#     shading=torch.div(norm_input,intensity)
#     shading=torch.clamp(shading,0.0,1.3)
#     shading=normalize_torch_0_1(shading)
#     shading=torch.where(
#         torch.abs(shading)<1e-4,
#         torch.Tensor([eps]).squeeze().cuda(),shading)
#     reflectance=torch.div(image,shading)
#     # print("reflectance",reflectance.shape)
#     # reflectance torch.Size([10, 3, 512, 512])
#     reflectance=torch.clamp(reflectance,0.0,1.2)
#     reflectance=normalize_torch_0_1(reflectance)
#     return reflectance

def colorize_tc(intensity,image,eps=1e-4):
    same_cuda=torch.device('cuda:'+str(intensity.get_device()))

    # batch,channel,H,W
    intensity=torch.where(
        torch.abs(intensity)<1e-4,
        torch.Tensor([eps]).squeeze().cuda(same_cuda),intensity)
    norm_input=torch.mean(image,dim=1,keepdim=True)
    # print("norm_input",norm_input.shape)
    # norm_input torch.Size([10, 1, 512, 512])
    
    shading=torch.div(norm_input,intensity)
    shading=torch.where(
        torch.abs(shading)<1e-4,
        torch.Tensor([eps]).squeeze().cuda(same_cuda),shading)
    
    reflectance=torch.div(image,shading)
    # print("reflectance",reflectance.shape)
    # reflectance torch.Size([10, 3, 512, 512])

    return reflectance

def rgb_to_srgb(rgb):
    ret = np.zeros_like(rgb)
    idx0 = rgb <= 0.0031308
    idx1 = rgb > 0.0031308
    ret[idx0] = rgb[idx0] * 12.92
    ret[idx1] = np.power(1.055 * rgb[idx1], 1.0 / 2.4) - 0.055
    return ret

# img=load_img("/mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/result/one_img/main_basic_gan_more_loss_direct_rgb_fix_v1_srgb/main_basic_gan_more_loss_direct_rgb_fix_v1_003_009.png")/255.0
# img=rgb_to_srgb(img)
# scipy.misc.imsave('main_basic_gan_more_loss_direct_rgb_fix_v1_003_009_srgb.png',img)

# utils_dir="/mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/utils"
# sys.path.insert(0,utils_dir)
# import utils_common as utils_common
# f_o_list=utils_common.get_file_list(
#     "/mnt/1T-5e7/mycodehtml/data_col/cv/IID_f_w_w/iiw-dataset/data/set1_0/train_100/*.png")
# f_r_list=utils_common.get_file_list(
#     "/mnt/1T-5e7/image/IID_results/tnestmeyer/set1_0__100/*.png")

# for o_path,r_path in zip(f_o_list,f_r_list):
#     # c o_i_fn: one image file name without extension
#     o_i_fn=r_path.split('/')[-1].replace(".png","")

#     ori_img=load_img(o_path)
#     ref_img=load_img(r_path)

#     guided_img=create_guided_img(ori_img,ref_img)

#     # c img_a_bi: image after bilateral filter
#     # img_a_bi=bilateral_f(ori_img,ref_img)
#     # scipy.misc.imsave(o_i_fn+"_bilateral2020.png", img_a_bi)

#     # img_a_dtf=DTF(guided_img,ref_img)
#     # scipy.misc.imsave(o_i_fn+"_gui_bref_dtf4040.png", img_a_dtf)

#     img_a_dtf=DTF(ori_img,ref_img)
#     scipy.misc.imsave(o_i_fn+"_ori_bref_dtf4040.png", img_a_dtf)

#     img_a_dtf=DTF(ori_img,ref_img,60,60)
#     scipy.misc.imsave(o_i_fn+"_ori_bref_dtf6060.png", img_a_dtf)

#     # ref,sha=colorize(ref_img,ori_img)
#     # ref=rgb_to_srgb(ref/255.0)
#     # scipy.misc.imsave(o_i_fn+"r_bi2020.png",ref)
#     # scipy.misc.imsave('54-r_tnestmeyer_bilateral_40_40_colorized_S.png', shading)

def pad_img(img,p_s):
    # c ori_img_shape: you get shape of original image
    ori_img_shape=img.shape
    # print("ori_img_shape",ori_img_shape)

    # c left_p: you get padding size in left
    left_p=p_s-(ori_img_shape[0]%p_s)
    # print("left_p",left_p)
    
    # c top_p: you get padding size in top
    top_p=p_s-(ori_img_shape[1]%p_s)
    # print("top_p",top_p)

    # c p_img: you get padded image
    p_img=np.lib.pad(img,((left_p,0),(top_p,0),(0,0)),'symmetric')

    return p_img

def pad_img_all_sides(img,p_s):
    p_img=np.lib.pad(img,((p_s,p_s),(p_s,p_s),(0,0)),'constant',constant_values=(0,0))
    return p_img

def extract_patches(t_image,p_s):
    pa_li=[]

    # You iterate as much as height of t_image, and step is p_s
    for width in range(0,t_image.shape[1],p_s):
        # You iterate as much as width of t_image, and step is p_s
        for height in range(0,t_image.shape[0],p_s):
            # c one_pa: you get one patch image
            one_pa=t_image[height:height+p_s,width:width+p_s,:]
            # You append one patch image into pa_li
            pa_li.append(one_pa)
    
    # c pa_np: you stack pa_li
    pa_np=np.stack(pa_li,axis=0)

    return pa_np

def gpu_torch_to_np(t_tensor):
    if t_tensor.shape[1]==3:
        np_arr=t_tensor.detach().cpu().numpy().squeeze().transpose(1,2,0)
        return np_arr
    elif t_tensor.shape[1]==1:
        np_arr=t_tensor.detach().cpu().numpy().squeeze()
        return np_arr
    
def np_normalize(img):
    # Standardization
    # img=(img-np.mean(img))/np.std(img)

    # Normalize [0,1]
    # img=(img-np.min(img))/np.ptp(img)

    # Normalize [0,255] as integer
    img=(255*(img-np.min(img))/np.ptp(img)).astype(int)

    # # Normalize [-1,1]
    # img=2*(img-np.min(img))/np.ptp(img)-1

    return img

def i_show_conv_deconv_plt(img,gray=False):
    if gray==True:
        out_test=img.squeeze()
        out_test=out_test.cpu()
        out_test=out_test.data.numpy()
        out_test=out_test.transpose((1,2,0))
        out_test=out_test[:,:,0]
        plt.imshow(out_test,cmap="gray")
        plt.show()
    else:
        out_test=img.squeeze()
        out_test=out_test.cpu()
        out_test=out_test.data.numpy()
        out_test=out_test.transpose((1,2,0))
        out_test=out_test[:,:,:]
        plt.imshow(out_test)
        plt.show()

def image_grad(tensor_img):
    ten=tensor_img
    # if ten.shape[1]==1:
    #     a=torch.Tensor(
    #         [[ 1, 0,-1],
    #          [ 2, 0,-2],
    #          [ 1, 0,-1]]).cuda()
    #     b=torch.Tensor(
    #         [[ 1, 2, 1],
    #          [ 0, 0, 0],
    #          [-1,-2,-1]]).cuda()
    #     conv1=nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1,bias=False).cuda()
    #     conv1.weight=nn.Parameter(a.float().unsqueeze(0).unsqueeze(0))    
    #     G_x=conv1(ten)
    #     conv2=nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1,bias=False).cuda()
    #     conv2.weight=nn.Parameter(b.float().unsqueeze(0).unsqueeze(0))    
    #     G_y=conv2(ten)
    #     G=torch.sqrt(torch.pow(G_x,2)+torch.pow(G_y,2))
    #     # G=torch.sqrt(torch.sum(torch.pow(G_x,2))+torch.sum(torch.pow(G_y,2)))
    #     return G
    # elif ten.shape[1]==3:
    #     a=torch.Tensor(
    #         [[ 1, 0,-1],
    #          [ 2, 0,-2],
    #          [ 1, 0,-1]]).cuda()
    #     b=torch.Tensor(
    #         [[ 1, 2, 1],
    #          [ 0, 0, 0],
    #          [-1,-2,-1]]).cuda()
    #     conv1=nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1,bias=False).cuda()
    #     conv1.weight=nn.Parameter(a.float().unsqueeze(0).unsqueeze(0).repeat(1,3,1,1))
    #     print("ten.shape",ten.shape)
    #     # ten.shape torch.Size([2, 3, 512, 512])
    #     G_x=conv1(ten)
    #     print("G_x.shape",G_x.shape)
    #     # G_x.shape torch.Size([2, 1, 512, 512])
    #     for i in range(G_x.shape[0]):
    #         G_x_t=G_x[i,:,:,:].detach().cpu().numpy().transpose(1,2,0)
    #         plt.imshow(G_x_t)
    #         plt.show()
    #     afaf
    #     conv2=nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1,bias=False).cuda()
    #     conv2.weight=nn.Parameter(b.float().unsqueeze(0).unsqueeze(0).repeat(1,3,1,1))
    #     G_y=conv2(ten)
    #     # G=torch.sqrt(torch.pow(G_x,2)+torch.pow(G_y,2))
    #     G=torch.sqrt(torch.sum(torch.pow(G_x,2))+torch.sum(torch.pow(G_y,2)))
    #     return G
    a=torch.Tensor(
        [[ 1, 0,-1],
         [ 2, 0,-2],
         [ 1, 0,-1]]).cuda()
    b=torch.Tensor(
        [[ 1, 2, 1],
         [ 0, 0, 0],
         [-1,-2,-1]]).cuda()
    conv1=nn.Conv2d(3,out_channels=3,kernel_size=3,stride=1,padding=1,bias=False).cuda()
    # print("a.float().unsqueeze(0).unsqueeze(0).repeat(1,3,1,1)",a.float().unsqueeze(0).unsqueeze(0).repeat(3,3,1,1).shape)
    # a.float().unsqueeze(0).unsqueeze(0).repeat(1,3,1,1) torch.Size([1, 3, 3, 3])
    conv1.weight=nn.Parameter(a.float().unsqueeze(0).unsqueeze(0).repeat(3,3,1,1))
    # print("ten.shape",ten.shape)
    # ten.shape torch.Size([2, 3, 512, 512])
    G_x=conv1(ten)
    # print("G_x.shape",G_x.shape)
    # G_x.shape torch.Size([2, 1, 512, 512])
    # for i in range(G_x.shape[0]):
    #     G_x_t=G_x[i,:,:,:].detach().cpu().numpy().transpose(1,2,0)
    #     plt.imshow(G_x_t)
    #     plt.show()

    conv2=nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1,bias=False).cuda()
    conv2.weight=nn.Parameter(b.float().unsqueeze(0).unsqueeze(0).repeat(1,3,1,1))
    G_y=conv2(ten)
    # G=torch.sqrt(torch.pow(G_x,2)+torch.pow(G_y,2))
    G=torch.sqrt(torch.sum(torch.pow(G_x,2))+torch.sum(torch.pow(G_y,2)))
    return G

def get_gradient_of_image(tensor_img):
    ten=tensor_img
    if ten.shape[1]==1:
        a=Variable(torch.Tensor(
            [[1, 0, -1],
             [2, 0, -2],
             [1, 0, -1]]).cuda(),requires_grad=False)
        a=a.float().unsqueeze(0).unsqueeze(0)
        G_x=F.conv2d(ten,a)
        # print("G_x",G_x.shape)

        b=Variable(torch.Tensor(
            [[ 1, 2, 1],
             [ 0, 0, 0],
             [-1,-2,-1]]).cuda(),requires_grad=False)
        b=b.float().unsqueeze(0).unsqueeze(0)
        G_y=F.conv2d(ten,b)
        # print("G_y",G_y.shape)
       
        return G_x,G_y

    elif ten.shape[1]==3:
        a=Variable(torch.Tensor(
            [[1, 0, -1],
             [2, 0, -2],
             [1, 0, -1]]).cuda(),requires_grad=False)
        a=a.float().unsqueeze(0).unsqueeze(0).repeat(3,3,1,1)
        G_x=F.conv2d(ten,a)
        # print("G_x",G_x.shape)

        b=Variable(torch.Tensor(
            [[ 1, 2, 1],
             [ 0, 0, 0],
             [-1,-2,-1]]).cuda(),requires_grad=False)
        b=b.float().unsqueeze(0).unsqueeze(0).repeat(3,3,1,1)
        G_y=F.conv2d(ten,b)
        # print("G_y",G_y.shape)

        return G_x,G_y

# def get_gradient_of_image(tensor_img):
#     ten=tensor_img
#     if ten.shape[1]==1:
#         a=torch.Tensor(
#             [[ 1, 0,-1],
#             [ 2, 0,-2],
#             [ 1, 0,-1]]).cuda()
#         b=torch.Tensor(
#             [[ 1, 2, 1],
#             [ 0, 0, 0],
#             [-1,-2,-1]]).cuda()
#         conv1=nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1,bias=False).cuda()
#         conv1.weight=nn.Parameter(a.float().unsqueeze(0).unsqueeze(0),requires_grad=False)
#         # print("ten",ten.shape)
#         # ten torch.Size([2, 1, 512, 512])
#         G_x=conv1(ten)
#         # print("G_x",G_x.shape)
#         # G_x torch.Size([2, 1, 512, 512])
#         # for i in range(G_x.shape[0]):
#         #     G_x_t=G_x[i,:,:,:].detach().cpu().numpy().squeeze()
#         #     plt.imshow(G_x_t,cmap='gray')
#         #     plt.show()
        
#         conv2=nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1,bias=False).cuda()
#         conv2.weight=nn.Parameter(b.float().unsqueeze(0).unsqueeze(0),requires_grad=False)
#         G_y=conv2(ten)
        
#         return G_x,G_y
#     elif ten.shape[1]==3:
#         a=torch.Tensor(
#             [[ 1, 0,-1],
#             [ 2, 0,-2],
#             [ 1, 0,-1]]).cuda()
#         b=torch.Tensor(
#             [[ 1, 2, 1],
#             [ 0, 0, 0],
#             [-1,-2,-1]]).cuda()
#         conv1=nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1,bias=False).cuda()
#         conv1.weight=nn.Parameter(a.float().unsqueeze(0).unsqueeze(0).repeat(3,3,1,1),requires_grad=False)
#         G_x=conv1(ten)

#         conv2=nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1,bias=False).cuda()
#         conv2.weight=nn.Parameter(b.float().unsqueeze(0).unsqueeze(0).repeat(3,3,1,1),requires_grad=False)
#         G_y=conv2(ten)

#         return G_x,G_y

def image_grad_x(tensor_img):
    ten=tensor_img
    if ten.shape[1]==1:
        a=torch.Tensor(
            [[ 1, 0,-1],
             [ 2, 0,-2],
             [ 1, 0,-1]]).cuda()
        conv1=nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1,bias=False).cuda()
        conv1.weight=nn.Parameter(a.float().unsqueeze(0).unsqueeze(0))    
        G_x=conv1(ten)
        return G_x
    elif ten.shape[1]==3:
        a=torch.Tensor(
            [[ 1, 0,-1],
             [ 2, 0,-2],
             [ 1, 0,-1]]).cuda()
        conv1=nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1,bias=False).cuda()
        conv1.weight=nn.Parameter(a.float().unsqueeze(0).unsqueeze(0).repeat(1,3,1,1))
        G_x=conv1(ten)
        return G_x

def image_grad_y(tensor_img):
    ten=tensor_img
    if ten.shape[1]==1:
        b=torch.Tensor(
            [[ 1, 2, 1],
             [ 0, 0, 0],
             [-1,-2,-1]]).cuda()
        conv2=nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1,bias=False).cuda()
        conv2.weight=nn.Parameter(b.float().unsqueeze(0).unsqueeze(0))    
        G_y=conv2(ten)
        return G_y
    elif ten.shape[1]==3:
        b=torch.Tensor(
            [[ 1, 2, 1],
             [ 0, 0, 0],
             [-1,-2,-1]]).cuda()
        conv2=nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1,bias=False).cuda()
        conv2.weight=nn.Parameter(b.float().unsqueeze(0).unsqueeze(0).repeat(1,3,1,1))
        G_y=conv2(ten)
        return G_y

def image_grad_by_subtraction(tensor_img):
    h_gradient=tensor_img[:,:,:,0:-2]-tensor_img[:,:,:,2:]
    h_gradient=torch.sum(torch.abs(h_gradient))

    v_gradient=tensor_img[:,:,0:-2,:]-tensor_img[:,:,2:,:]
    v_gradient=torch.sum(torch.abs(v_gradient))

    gradient=h_gradient+v_gradient

    return gradient

def image_grad_sobel_opencv(img_np):
    stack_imgs=[]
    for one_img in range(img_np.shape[0]):
        one_img_np=img_np[one_img,:,:,:]

        G_x=cv2.Sobel(one_img_np,cv2.CV_64F,1,0,ksize=5)
        G_y=cv2.Sobel(one_img_np,cv2.CV_64F,0,1,ksize=5)
        G=np.sqrt(G_x**2+G_y**2)

        stack_imgs.append(G)

    out=np.stack(stack_imgs,axis=0)
    return out

def image_grad_convolve_scipy(img_np):
    a=np.array([[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]])
    b=np.array([[-1,-2,-1],
                [ 0, 0, 0],
                [ 1, 2, 1]])

    stack_imgs=[]
    for one_img in range(img_np.shape[0]):
        one_img_np=img_np[one_img,:,:,:]

        G_x=convolve(one_img_np,a)
        G_y=convolve(one_img_np,b)
        G=np.sqrt(G_x**2+G_y**2)

        stack_imgs.append(G)

    out=np.stack(stack_imgs,axis=0)
    
    return out

def norm_img_cv(img):
    norm_img=cv2.normalize(img,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
    return norm_img

def normalize_torch_0_1(img):
    min_v=torch.min(img)
    range_v=torch.max(img)-min_v
    if range_v>0:
        img=(img-min_v)/range_v
    else:
        img=torch.zeros(img.size())
    return img

def get_sha_img(O_img_tc,R_gt_img_tc):
    # same_cuda=torch.device('cuda:'+str(O_img_tc.get_device()))

    # O_img_tc=torch.where(
    #     O_img_tc<1e-4,
    #     torch.Tensor([1e-4]).squeeze().cuda(),O_img_tc)
    # R_gt_img_tc=torch.where(
    #     torch.abs(R_gt_img_tc)<1e-4,
    #     torch.Tensor([1e-4]).squeeze().cuda(same_cuda),R_gt_img_tc)

    R_gt_img_tc=torch.where(
      torch.abs(R_gt_img_tc)<1e-4,
      torch.cuda.FloatTensor([1e-4]).squeeze(),R_gt_img_tc)
    
    print("Before")
    print("O_img_tc.get_device()",O_img_tc.get_device())
    print("R_gt_img_tc.get_device()",R_gt_img_tc.get_device())
    print("R_gt_img_tc",R_gt_img_tc)
    
    # O_img_tc.cuda(R_gt_img_tc.get_device())

    print("After")
    print("O_img_tc.get_device()",O_img_tc.get_device())
    print("R_gt_img_tc.get_device()",R_gt_img_tc.get_device())
    print("R_gt_img_tc",R_gt_img_tc)


    sha=torch.clamp(torch.div(O_img_tc,R_gt_img_tc),0.0,1.3)[:,0,:,:].unsqueeze(1)
    # print("sha",sha.shape)
    # sha torch.Size([10, 1, 512, 512])
    sha=normalize_torch_0_1(sha)

    # sha=denoise_tv_chambolle(sha,weight=0.1,multichannel=True)

    # for i in range(sha.shape[0]):
    #     plt.imshow(sha[i,0,:,:].detach().cpu().numpy(),cmap="gray")
    #     plt.show()

    return sha

def get_ref_img(dense_O_img_tc,dense_S_gt_img_tc):
    # same_cuda=torch.device('cuda:'+str(dense_O_img_tc.get_device()))

    # print("dense_S_gt_img_tc",dense_S_gt_img_tc.shape)
    # dense_S_gt_img_tc torch.Size([4, 1, 1024, 1024])
    dense_S_gt_img_tc=dense_S_gt_img_tc.repeat(1,3,1,1)
    # print("dense_S_gt_img_tc",dense_S_gt_img_tc.shape)
    # dense_S_gt_img_tc torch.Size([4, 3, 1024, 1024])

    # min_v=torch.min(dense_S_gt_img_tc)
    # range_v=torch.max(dense_S_gt_img_tc)-min_v
    # if range_v>0:
    #     dense_S_gt_img_tc=(dense_S_gt_img_tc-min_v)/range_v
    # else:
    #     dense_S_gt_img_tc=torch.zeros(dense_S_gt_img_tc.size())
    
    # print("dense_S_gt_img_tc",torch.min(dense_S_gt_img_tc))
    # print("dense_S_gt_img_tc",torch.max(dense_S_gt_img_tc))
    # print("dense_O_img_tc",torch.min(dense_O_img_tc))
    # print("dense_O_img_tc",torch.max(dense_O_img_tc))

    # dense_O_img_tc=torch.where(
    #     dense_O_img_tc<1e-4,
    #     torch.Tensor([1e-4]).squeeze().cuda(),dense_O_img_tc)
    # dense_S_gt_img_tc=torch.where(
    #     torch.abs(dense_S_gt_img_tc)<1e-4,
    #     torch.Tensor([1e-4]).squeeze().cuda(same_cuda),dense_S_gt_img_tc)
    dense_S_gt_img_tc=torch.where(
        torch.abs(dense_S_gt_img_tc)<1e-4,
        torch.Tensor([1e-4]).squeeze().cuda(),dense_S_gt_img_tc)

    print("Before")
    print("dense_O_img_tc.get_device()",dense_O_img_tc.get_device())
    print("dense_S_gt_img_tc.get_device()",dense_S_gt_img_tc.get_device())
    
    R_gt_img_tc=O_img_tc.new(R_gt_img_tc)
    print("After")
    print("dense_O_img_tc.get_device()",dense_O_img_tc.get_device())
    print("dense_S_gt_img_tc.get_device()",dense_S_gt_img_tc.get_device())
    
    ref=torch.div(dense_O_img_tc,dense_S_gt_img_tc)
    # ref=torch.clamp(torch.div(dense_O_img_tc,dense_S_gt_img_tc),0.0,1.2)
    # print("ref",ref.shape)
    # ref torch.Size([10, 3, 512, 512])
    
    # ref=normalize_torch_0_1(ref)

    # for i in range(ref.shape[0]):
    #     plt.imshow(ref[i,:,:,:].detach().cpu().numpy().transpose(1,2,0))
    #     plt.show()
    
    return ref

def remove_0_in_img(img):
    img[img==0.0]=0.0001
    return img

def create_sha_img(ori_img,ref_img):
    ori_img=ori_img/255.0
    ref_img=ref_img/255.0

    ori_img[ori_img==0.0]=0.0001
    ref_img[ref_img==0.0]=0.0001

    sha=ori_img/ref_img
    # sha=np.nan_to_num(ori_img/ref_img)

    sha=np.clip(sha,0,3)

    sha=cv2.normalize(sha,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)[:,:,0].astype("float16")

    return sha

def gray_to_rgb(gray):
    rgb = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=gray.dtype)
    rgb[:, :, :] = gray[:, :, np.newaxis]
    return rgb

def luminance(image):
    """ Returns the luminance image """
    if image.ndim == 2:
        return np.dot(RGB_TO_Y, image.T).T
    else:
        rows, cols, _ = image.shape
        image_flat = image.reshape(rows * cols, 3)
        Y_flat = np.dot(RGB_TO_Y, image_flat.T).T
        return Y_flat.reshape(image.shape[0:2])

def rescale_for_display(image, mask_nz=None, percentile=99.9):
    """ Rescales an image so that a particular perenctile is mapped to pure
    white """
    if mask_nz is not None:
        return image / np.percentile(image, percentile)
    else:
        return image / np.percentile(image[mask_nz], percentile)

def rgb_to_srgb(rgb):
    ret = np.zeros_like(rgb)
    idx0 = rgb <= 0.0031308
    idx1 = rgb > 0.0031308
    ret[idx0] = rgb[idx0] * 12.92
    ret[idx1] = np.power(1.055 * rgb[idx1], 1.0 / 2.4) - 0.055
    return ret

def srgb_to_rgb(srgb):
    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret

def rgb_to_irg(rgb):
    """ converts rgb to (mean of channels, red chromaticity, green chromaticity) """
    irg=np.zeros_like(rgb)
    s=np.sum(rgb,axis=-1)
    irg[...,0]=s/3.0
    irg[...,1]=rgb[...,0]/s
    irg[...,2]=rgb[...,1]/s
    return irg

def torch_rgb_to_torch_irg(torch_rgb):
    """ converts rgb to (mean of channels, red chromaticity, green chromaticity) """
    # same_cuda=torch.device('cuda:'+str(torch_rgb.get_device()))

    # irg=torch.zeros(torch_rgb.shape).cuda(same_cuda)
    irg=torch.zeros(torch_rgb.shape).cuda()
    # print("irg",irg.shape)
    # irg torch.Size([2, 3, 528, 528])

    s=torch.sum(torch_rgb,dim=1)

    # irg[:,0,:,:]=torch.div(s,torch.Tensor([3.0]).cuda(same_cuda))
    irg[:,0,:,:]=torch.div(s,torch.Tensor([3.0]).cuda())
    irg[:,1,:,:]=torch.div(torch_rgb[:,0,:,:],s)
    irg[:,2,:,:]=torch.div(torch_rgb[:,1,:,:],s)
    # print("irg",irg.shape)
    # irg torch.Size([2, 3, 528, 528])

    return irg

def torch_rgb_to_torch_irg_full(torch_rgb):
    """ converts rgb to (mean of channels, red chromaticity, green chromaticity) """
    # same_cuda=torch.device('cuda:'+str(torch_rgb.get_device()))

    # irg=torch.zeros(torch_rgb.shape).cuda(same_cuda)
    irg=torch.zeros(torch_rgb.shape).cuda()
    # print("irg",irg.shape)
    # irg torch.Size([2, 3, 528, 528])

    s=torch.sum(torch_rgb,dim=1)

    irg[:,0,:,:]=torch.div(torch_rgb[:,0,:,:],s)
    irg[:,1,:,:]=torch.div(torch_rgb[:,1,:,:],s)
    irg[:,2,:,:]=torch.div(torch_rgb[:,2,:,:],s)
    # print("irg",irg.shape)
    # irg torch.Size([2, 3, 528, 528])

    return irg

def irg_to_rgb(irg):
    """ converts (mean of channels, red chromaticity, green chromaticity) to rgb """
    rgb = np.zeros_like(irg)
    s = irg[..., 0] * 3.0
    rgb[..., 0] = irg[..., 1] * s
    rgb[..., 1] = irg[..., 2] * s
    rgb[..., 2] = (1.0 - irg[..., 1] - irg[..., 2]) * s

    #np.testing.assert_array_almost_equal(
        #irg, rgb_to_irg(rgb))

    return rgb

def load_np_file(file_p):
    loaded=np.load(file_p)
    
    images=loaded['images']/255.0
    # print("images",images.shape)
    # images (100, 3, 1024, 1024)

    comparisons=loaded['comparisons']
    # print("comparisons",comparisons.shape)
    # comparisons (100, 1182, 1, 6)

    # {"1":1,"2":2,"E":0}
    # [point1,point2,darker_num,darker_score]

    return images,comparisons

def preprocess_dense_imgs(ori_img,srgb_gt_r_img,fn):
    """
    Act
      Use for CGINTRINSIC, MIT IID which has many 0 pixel (black) values
    """
    # c srgb_img: actually rgb original image than srgb original image
    srgb_img=ori_img
    file_name=fn

    gt_R=srgb_gt_r_img

    mask=np.ones((srgb_gt_r_img.shape[0],srgb_gt_r_img.shape[1]))
    # print("mask",mask.shape)
    # mask (341, 512)

    # c gt_R_gray: mean of R gt image
    gt_R_gray = np.mean(gt_R, 2)
    mask[gt_R_gray < 1e-6] = 0
    # mean of original image
    mask[np.mean(srgb_img,2) < 1e-6] = 0 

    # plt.imshow(mask,cmap="gray")
    # plt.show()
    mask = skimage.morphology.binary_erosion(mask, square(11))
    mask = np.expand_dims(mask, axis = 2)
    mask = np.repeat(mask, 3, axis= 2)
    gt_R[gt_R <1e-6] = 1e-6

    rgb_img = srgb_img
    gt_S = rgb_img / gt_R
    # plt.imshow(gt_S[:,:,0],cmap='gray')
    # plt.show()

    # search_name = path[:-4] + ".rgbe"
    # irridiance = self.stat_dict[search_name]

    # if irridiance < 0.25:
    #     srgb_img = denoise_tv_chambolle(srgb_img, weight=0.05, multichannel=True)
    #     gt_S = denoise_tv_chambolle(gt_S, weight=0.1, multichannel=True)

    mask[gt_S > 10] = 0
    gt_S[gt_S > 20] = 20
    mask[gt_S < 1e-4] = 0
    gt_S[gt_S < 1e-4] = 1e-4

    if np.sum(mask) < 10:
        max_S = 1.0
    else:
        max_S = np.percentile(gt_S[mask > 0.5], 90)

    gt_S = gt_S/max_S

    gt_S = np.mean(gt_S, 2)
    gt_S = np.expand_dims(gt_S, axis = 2)

    gt_R = np.mean(gt_R,2)
    gt_R = np.expand_dims(gt_R, axis = 2)

    # print("srgb_img",srgb_img.shape)
    # print("gt_R",gt_R.shape)
    # print("gt_S",gt_S.shape)
    # print("mask",mask.shape)
    # srgb_img (341, 512, 3)
    # gt_R (341, 512, 1)
    # gt_S (341, 512, 1)
    # mask (341, 512, 3)

    # plt.imshow(srgb_img)
    # plt.show()
    # plt.imshow(gt_R)
    # plt.show()
    # plt.imshow(gt_S[:,:,0],cmap="gray")
    # plt.show()
    # plt.imshow(mask[:,:,0],cmap="gray")
    # plt.show()
    
    ori_3c_img=srgb_img
    gt_R_1c_img=gt_R
    gt_S_1c_img=gt_S
    mask_3c_img=mask

    return ori_3c_img, gt_R_1c_img, gt_S_1c_img, mask_3c_img

def detect_img_less_than_256(txt_file_containing_paths_of_img):
  """
  Act
    * 
  
  Params
    * txt_file_containing_paths_of_img
    "/mnt/1T-5e7/image/whole_dataset/text_for_colab/real/temp/bigtime_trn.txt"
  
  Return
    * have_small_img
    If 0, you have no small image
    * small_lens
    [199,200] means you have images which has 199 or 200 lengths
  """
  p_l_full,num_img=utils_common.return_path_list_from_txt(txt_file_containing_paths_of_img)
  # print("num_img",num_img)
  # 8400

  # c col_i_szs: collection of image sizes
  col_i_szs=[]
  # c col_i_pa_szs: collection of image paths and sizes
  col_i_pa_szs=[]
  for one_path in p_l_full:
    one_path=one_path.replace("\n","")
    one_lo_img=utils_image.load_img(one_path)

    col_i_szs.append(one_lo_img.shape[:2])
    col_i_pa_szs.append([one_path,one_lo_img.shape[:2]])
  
  col_i_szs=list(set(col_i_szs))

  col_i_szs_np=np.array(col_i_szs)

  # c have_small_img: you have small images?
  have_small_img=np.sum(col_i_szs_np<256)
  # print("have_small_img",have_small_img)

  # c small_lens: small length
  small_lens=col_i_szs_np[col_i_szs_np<256]

  return have_small_img,small_lens

# ================================================================================
# for i in range(irg_imgs.shape[0]):
#     plt.imshow(irg_imgs[i,:,:,:].detach().cpu().numpy().transpose(1,2,0))
#     plt.show()

# for i in range(sha.shape[0]):
#     plt.imshow(sha[i,0,:,:].detach().cpu().numpy(),cmap="gray")
#     plt.show()

# for i in range(sampled_rgt_imgs.shape[0]):
#     plt.subplot(1,2,1)
#     plt.imshow(sampled_trn_imgs[i,:,:,:])
#     plt.subplot(1,2,2)
#     plt.imshow(sampled_rgt_imgs[i,:,:,:])
#     plt.show()
