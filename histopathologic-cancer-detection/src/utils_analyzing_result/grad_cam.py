import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse
import os
import copy

from src.utils import utils_image as utils_image

i=0
# resnet = models.resnet50(pretrained=True)
image = []

class FeatureExtractor():
    """
    Class for extracting activations and 
    registering gradients from targetted intermediate layers
    """
    def __init__(self, model, target_layers):
        self.model=model
        self.target_layers=target_layers
        self.gradients=[]

    def save_gradient(self, grad):
      self.gradients.append(grad)

    def __call__(self,x):
      outputs=[]
      self.gradients=[]

      # Iterates all modules
      for name,module in self.model._modules.items():
        # print("name",name)
        # conv1
        
        # print("module",module)
        # Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # ================================================================================
        x=module(x)
        # print("x after",x.grad_fn)
        # torch.Size([1, 64, 48, 48])

        # ================================================================================
        # If module name is in target ["layer4"]
        if name in self.target_layers:
          # Call hook function to save gradients with respect to output prediction x
          x.register_hook(self.save_gradient)

          # And you also save output prediction x itself into outputs
          outputs+=[x]

      return outputs,x

class ModelOutputs():
  """
  Class for making a forward pass, and getting:
  1. The network output.
  2. Activations from intermeddiate targetted layers.
  3. Gradients from intermeddiate targetted layers. """
  def __init__(self, model,model_with_fc,target_layers):
    self.model=model
    self.model_with_fc=model_with_fc
    self.feature_extractor = FeatureExtractor(self.model, target_layers)

  def get_gradients(self):
    return self.feature_extractor.gradients

  def __call__(self, x):
    target_activations,output=self.feature_extractor(x)
    # print("target_activations",len(target_activations))
    # print("output",output.shape)
    # 1
    # torch.Size([1, 2048, 1, 1])

    # c output: flatten output
    output=output.view(output.size(0),-1)
    # print("output",output.shape)
    # torch.Size([1, 147456])
    
    # Pass output into fc layer of model
    output=self.model_with_fc.fc(output)
    # print("output",output.shape)
    # torch.Size([1, 2])

    return target_activations, output

def preprocess_image(img):
  means=[0.485, 0.456, 0.406]
  stds=[0.229, 0.224, 0.225]

  preprocessed_img = img.copy()

  for i in range(3):
    preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
    preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
  preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
  preprocessed_img = torch.from_numpy(preprocessed_img)
  preprocessed_img.unsqueeze_(0)

  input=Variable(preprocessed_img,requires_grad=True)
  # print("input",input.shape)
  # torch.Size([1, 3, 96, 96])

  return input

def show_cam_on_image(img, mask,name):
  # print("img",img)
  # print("mask",mask)
  # print("name",name)
  # img /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/test/ffcaef8b9006b4d0b128328e6df6e4d139d3c40a.tif
  # mask [[0.30953777 0.30953777 0.30953777 ... 0.31883815 0.31883815 0.31883815]
  # [0.30953777 0.30953777 0.30953777 ... 0.31883815 0.31883815 0.31883815]
  # [0.30953777 0.30953777 0.30953777 ... 0.31883815 0.31883815 0.31883815]
  # ...
  # [0.         0.         0.         ... 0.15856665 0.15856665 0.15856665]
  # [0.         0.         0.         ... 0.15856665 0.15856665 0.15856665]
  # [0.         0.         0.         ... 0.15856665 0.15856665 0.15856665]]
  # name 1





  heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
  heatmap = np.float32(heatmap) / 255
  cam = heatmap + np.float32(img)
  cam = cam / np.max(cam)
  # print("cam",cam.shape)
  # cam (96, 96, 3)
  import scipy.misc
  scipy.misc.imsave('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/my_model/prj_root/src/utils_for_analyze/cam.png',cam)
  # cv2.imwrite("./cam_{}.jpg".format(name), np.uint8(255 * cam))
  aafaf

class GradCam:
  def __init__(self, model,model_with_fc,target_layer_names):
    
    self.model=model
    self.model.eval()
    self.model.cuda()
    
    self.model_with_fc=model_with_fc
    self.model_with_fc.eval()
    self.model_with_fc.cuda()

    self.extractor = ModelOutputs(self.model,self.model_with_fc,target_layer_names)

  def forward(self, input):
    return self.model(input) 

  def __call__(self, input, index = None):
    features, output = self.extractor(input.cuda())
    # print(features)
    # print("output",output.shape)
  
    if index==None:
      index=np.argmax(output.cpu().data.numpy())
      # print("index",index)
      # 1

    one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
    one_hot[0][index] = 1
    one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)

    one_hot = torch.sum(one_hot.cuda() * output)

    self.model.zero_grad()
    one_hot.backward(retain_graph=True)

    grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

    target = features[-1]
    target = target.cpu().data.numpy()[0, :]

    weights = np.mean(grads_val, axis = (2, 3))[0, :]
    cam = np.zeros(target.shape[1 : ], dtype = np.float32)

    for i, w in enumerate(weights):
      cam += w * target[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (96,96))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    return cam

class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
  def __init__(self, model, use_cuda):
    self.model = resnet
    self.model.eval()
    self.cuda = use_cuda
    if self.cuda:
      self.model = model.cuda()

    # replace ReLU with GuidedBackpropReLU
    for idx, module in self.model._modules.items():
      if module.__class__.__name__ == 'ReLU':
        self.model._modules[idx] = GuidedBackpropReLU()

  def forward(self, input):
    return self.model(input)

  def __call__(self, input, index = None):
    if self.cuda:
      output = self.forward(input.cuda())
    else:
      output = self.forward(input)

    if index == None:
      index = np.argmax(output.cpu().data.numpy())

    one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
    one_hot[0][index] = 1
    one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
    if self.cuda:
      one_hot = torch.sum(one_hot.cuda() * output)
    else:
      one_hot = torch.sum(one_hot * output)

    # self.model.features.zero_grad()
    # self.model.classifier.zero_grad()
    one_hot.backward(retain_graph=True)

    output = input.grad.cpu().data.numpy()
    output = output[0,:,:,:]

    return output

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--use-cuda', action='store_true', default=False,
                      help='Use NVIDIA GPU acceleration')
  parser.add_argument('--image-path', type=str, default='./examples/',
                      help='Input image path')
  args = parser.parse_args()
  args.use_cuda = args.use_cuda and torch.cuda.is_available()
  if args.use_cuda:
      print("Using GPU for acceleration")
  else:
      print("Using CPU for computation")

  return args

def initialize_grad_cam(model,list_of_img_paths,args):
  del model.softmax_layer
  model=model.resnet50
  model_with_fc=copy.deepcopy(model)
  del model.fc
  # print("model",model)
  # region 
  # ResNet(
  #   (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  #   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #   (relu): ReLU(inplace)
  #   (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  #   (layer1): Sequential(
  #     (0): Bottleneck(
  #       (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  #       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (relu): ReLU(inplace)
  #       (downsample): Sequential(
  #         (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #         (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       )
  #     )
  #     (1): Bottleneck(
  #       (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  #       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (relu): ReLU(inplace)
  #     )
  #     (2): Bottleneck(
  #       (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  #       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (relu): ReLU(inplace)
  #     )
  #   )
  #   (layer2): Sequential(
  #     (0): Bottleneck(
  #       (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  #       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (relu): ReLU(inplace)
  #       (downsample): Sequential(
  #         (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
  #         (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       )
  #     )
  #     (1): Bottleneck(
  #       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  #       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (relu): ReLU(inplace)
  #     )
  #     (2): Bottleneck(
  #       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  #       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (relu): ReLU(inplace)
  #     )
  #     (3): Bottleneck(
  #       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  #       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (relu): ReLU(inplace)
  #     )
  #   )
  #   (layer3): Sequential(
  #     (0): Bottleneck(
  #       (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (relu): ReLU(inplace)
  #       (downsample): Sequential(
  #         (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
  #         (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       )
  #     )
  #     (1): Bottleneck(
  #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (relu): ReLU(inplace)
  #     )
  #     (2): Bottleneck(
  #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (relu): ReLU(inplace)
  #     )
  #     (3): Bottleneck(
  #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (relu): ReLU(inplace)
  #     )
  #     (4): Bottleneck(
  #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (relu): ReLU(inplace)
  #     )
  #     (5): Bottleneck(
  #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (relu): ReLU(inplace)
  #     )
  #   )
  #   (layer4): Sequential(
  #     (0): Bottleneck(
  #       (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  #       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (relu): ReLU(inplace)
  #       (downsample): Sequential(
  #         (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
  #         (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       )
  #     )
  #     (1): Bottleneck(
  #       (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  #       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (relu): ReLU(inplace)
  #     )
  #     (2): Bottleneck(
  #       (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  #       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #       (relu): ReLU(inplace)
  #     )
  #   )
  #   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  #   (fc): Linear(in_features=2048, out_features=2, bias=True)
  # )
  # endregion 
 
  # ================================================================================
  grad_cam=GradCam(model,model_with_fc,target_layer_names=["layer4"])

  # ================================================================================
  for img_p in list_of_img_paths:
    loaded_img=utils_image.load_img(img_p)/255.0
    loaded_img=loaded_img.astype("float32")
    
    input=preprocess_image(loaded_img)
    # <class 'torch.Tensor'>
    
    target_index=None

    mask = grad_cam(input,target_index)
    # afaf 1: mask = grad_cam(input, target_index)
    i=0
    i=i+1 

    show_cam_on_image(loaded_img, mask,i)
  afaaf  