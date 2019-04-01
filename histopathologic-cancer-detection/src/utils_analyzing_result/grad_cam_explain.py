# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/my_model/prj_root/src/utils_for_analyze && \
# rm e.l && python grad_cam_explain.py \
# 2>&1 | tee -a e.l && code e.l


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

sys.path.insert(0,"/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/my_model/prj_root")
from src.utils import utils_image as utils_image

i=0
resnet = models.resnet50(pretrained=True)
image = []

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
      self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
          # print("name",name)
          # print("module",module)
          # conv1
          # Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
          x = module(x)
          print("x",x.grad_fn)
          if name in self.target_layers:
              x.register_hook(self.save_gradient)
              outputs += [x]
        return outputs, x

class ModelOutputs():
  """ Class for making a forward pass, and getting:
  1. The network output.
  2. Activations from intermeddiate targetted layers.
  3. Gradients from intermeddiate targetted layers. """
  def __init__(self, model, target_layers):
    self.model = model
    self.feature_extractor = FeatureExtractor(self.model, target_layers)

  def get_gradients(self):
    return self.feature_extractor.gradients

  def __call__(self, x):
    target_activations, output  = self.feature_extractor(x)
    print("target_activations",len(target_activations))
    print("output",output.shape)
    afaf
    
    output = output.view(output.size(0), -1)
    output = resnet.fc(output)
    #print(output.size())
    return target_activations, output

def preprocess_image(img):
  means=[0.485, 0.456, 0.406]
  stds=[0.229, 0.224, 0.225]

  preprocessed_img = img.copy()[: , :, ::-1]
  for i in range(3):
    preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
    preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
  preprocessed_img = \
    np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
  preprocessed_img = torch.from_numpy(preprocessed_img)
  preprocessed_img.unsqueeze_(0)
  input = Variable(preprocessed_img, requires_grad = True)
  return input

def show_cam_on_image(img, mask,name):
  heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
  heatmap = np.float32(heatmap) / 255
  cam = heatmap + np.float32(img)
  cam = cam / np.max(cam)
  cv2.imwrite("./cam_{}.jpg".format(name), np.uint8(255 * cam))

class GradCam:
  def __init__(self, model, target_layer_names, use_cuda):
    self.model = model
    self.model.eval()
    self.cuda = use_cuda
    if self.cuda:
      self.model = model.cuda()

    self.extractor = ModelOutputs(self.model, target_layer_names)

  def forward(self, input):
    return self.model(input) 

  def __call__(self, input, index = None):
    if self.cuda:
      features, output = self.extractor(input.cuda())
    else:
      features, output = self.extractor(input)

    if index == None:
      index = np.argmax(output.cpu().data.numpy())

    one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
    one_hot[0][index] = 1
    one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
    if self.cuda:
      one_hot = torch.sum(one_hot.cuda() * output)
    else:
      one_hot = torch.sum(one_hot * output)

    self.model.zero_grad()##这两行同理，features不包含，可以重新加回去试一试，会报错不包含这个对象。
    self.model.zero_grad()
    one_hot.backward(retain_graph=True)##这里适配我们的torch0.4及以上，我用的1.0也可以完美兼容。（variable改成graph即可）

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

if __name__ == '__main__':
  """ python grad_cam.py <path_to_image>
  1. Loads an image with opencv.
  2. Preprocesses it for VGG19 and converts to a pytorch variable.
  3. Makes a forward pass to find the category index with the highest score,
  and computes intermediate activations.
  Makes the visualization. """

  args = get_args()

  # Can work with any model, but it assumes that the model has a 
  # feature method, and a classifier method,
  # as in the VGG models in torchvision.
  model = models.resnet50(pretrained=True)
  del model.fc
  print(model)
  #modules = list(resnet.children())[:-1]
  #model = torch.nn.Sequential(*modules)

  # print(model)
  grad_cam = GradCam(model, target_layer_names = ["layer4"], use_cuda=args.use_cuda)
  # x=os.walk(args.image_path)  
  # for root,dirs,filename in x:
  #   # print(type(grad_cam))
  #   print(filename)
  
  # for s in filename:
  #     image.append(cv2.imread(args.image_path+s,1))
    #img = cv2.imread(filename, 1)

  image=[
    "/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/test/ffcaef8b9006b4d0b128328e6df6e4d139d3c40a.tif",
    "/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/test/ffcc29cf0e363737b577d1db470df0bb1adf7957.tif"]  
  for img in image:
    img=utils_image.load_img(img)/255.0
    img=img.astype("float32")
    # img = np.float32(cv2.resize(img, (224, 224))) / 255
    input = preprocess_image(img)
    # print("input",input.grad_fn)
    # None

    print('input.size()=',input.size())
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index =None

    mask = grad_cam(input, target_index)
    # print(type(mask))
    i=i+1 
    show_cam_on_image(img, mask,i)

    #gb_model = GuidedBackpropReLUModel(model = model, use_cuda=args.use_cuda)
    #gb = gb_model(input, index=target_index)
    #utils.save_image(torch.from_numpy(gb), 'gb.jpg')

    #cam_mask = np.zeros(gb.shape)
    #for i in range(0, gb.shape[0]):
        #  cam_mask[i, :, :] = mask

    #cam_gb = np.multiply(cam_mask, gb)
    #utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')