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

      last_layer=list(self.model.layer4.children())[-1]

      # Iterates all modules
      for name,module in self.model._modules.items():

        x=module(x)

        # ================================================================================
        if name in self.target_layers:
          
          for name_la,module_la in last_layer._modules.items():
            if name_la in "conv3":
              # Call hook function to save gradients with respect to output prediction x
              x.register_hook(self.save_gradient)

              # And you also save output prediction x itself into outputs
              outputs+=[x]
            else:
              print("passed")

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
    grad_out=self.feature_extractor.gradients
    # print("grad_out",grad_out[0].shape)
    # torch.Size([1, 2048, 1, 1])

    return grad_out

  def __call__(self, x):
    target_activations,output=self.feature_extractor(x)
    # print("target_activations",target_activations[0].shape)
    # print("output",output.shape)
    # torch.Size([1, 2048, 1, 1])
    # torch.Size([1, 2048, 1, 1])

    # c output: flatten output
    output=output.view(output.size(0),-1)
    # print("output",output.shape)
    # torch.Size([1, 147456])
    
    # Pass output into fc layer of model
    output=self.model_with_fc.fc(output)
    # print("output",output.shape)
    # torch.Size([1, 2])
    # torch.Size([1, 1])

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

def show_cam_on_image(img,mask,name):
  # print("img",img.shape)
  # (224, 224, 3)
  # Original image

  heatmap = cv2.applyColorMap(np.uint8(255*mask),cv2.COLORMAP_JET)
  # print("heatmap",heatmap.shape)
  # (224, 224, 3)
  
  heatmap=heatmap[:,:,::-1]

  heatmap = np.float32(heatmap) / 255
  # print("heatmap",heatmap.shape)
  # (224, 224, 3)

  cam = heatmap + np.float32(img)
  cam = cam / np.max(cam)
  # print("cam",cam.shape)
  # cam (96, 96, 3)
  import scipy.misc
  scipy.misc.imsave('/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/my_model/my_model_new_architecture/prj_root/src/utils_analyzing_result/cam.png',cam)
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
  
    if index==None:
      index=np.argmax(output.cpu().data.numpy())

    one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)

    one_hot[0][index] = 1
    # one_hot[0][index] = 0

    one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)

    one_hot = torch.sum(one_hot.cuda() * output)
    # print("one_hot",one_hot)
    # tensor(-28.1434, device='cuda:0', grad_fn=<SumBackward0>)

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
    cam = cv2.resize(cam, (224,224))
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

def initialize_grad_cam(model,list_of_img_paths,args):
  
  # @ Get your trained network
  # print("model",model)

  # ================================================================================
  # @ Deep copy your trained network having FC
  model_with_fc=copy.deepcopy(model)

  # model = models.resnet50(pretrained=True)
  # del model.fc
  # print(model)

  # del model_with_fc.fc
  # del list(model_with_fc.layer4.children())[-1].cbam
  

  # ================================================================================
  # @ Delete last fully connected layer
  del model.fc
  # del list(model.layer4.children())[-1].cbam
  # print("model",list(model.layer4.children())[-1])
  # print("model",dir(model))
  # print("model",dir(model.layer4))
  # print("model",model)

  # ================================================================================
  grad_cam=GradCam(model,model_with_fc,target_layer_names=["layer1"])

  # ================================================================================
  for img_p in list_of_img_paths:
    loaded_img=utils_image.load_img(img_p)/255.0
    loaded_img=utils_image.resize(loaded_img,(224,224))
    loaded_img=loaded_img.astype("float32")
    
    input=preprocess_image(loaded_img)
    # <class 'torch.Tensor'>
    
    target_index=None

    mask = grad_cam(input,target_index)
    import scipy.misc
    # scipy.misc.imsave('./mask.png',mask)
    # afaf 1: mask = grad_cam(input, target_index)
    i=0
    i=i+1 

    show_cam_on_image(loaded_img, mask,i)
