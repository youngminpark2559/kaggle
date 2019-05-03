# @ Basic modules
import sys,os,copy,argparse

import torch

# ================================================================================
from src.utils import utils_net as utils_net

from src.networks import networks as networks

from src.utils import utils_for_custom_optim_and_lr as utils_for_custom_optim_and_lr

# ================================================================================
class Model_API_class():
  def __init__(self,args):
    super(Model_API_class).__init__()

    # ================================================================================
    # @ Configure member variables
    self.lr=0.0001
    self.weight_decay=0.1
    self.args=args
    
    if self.args.network_type=="Scikit_Learn_SVM":
      self.gen_net=self.net_generator()
    else:
      self.gen_net,self.optimizer=self.net_generator()

  def net_generator(self):

    if self.args.train_method=="train_by_transfer_learning_using_resnet":
      
      # ================================================================================
      # @ Create network

      if self.args.network_type=="Pretrained_ResNet152":
        gen_net=networks.Pretrained_ResNet152().cuda()
      elif self.args.network_type=="Pretrained_ResNet50":
        gen_net=networks.Pretrained_ResNet50().cuda()
      elif self.args.network_type=="Pretrained_VGG16":
        gen_net=networks.Pretrained_VGG16().cuda()
      elif self.args.network_type=="Custom_Net":
        gen_net=networks.Custom_Net().cuda()
      elif self.args.network_type=="ResNet50_CBAM":
        gen_net=networks.ResNet50_CBAM(layers=[3,4,6,3],network_type="Pathology",num_classes=28,att_type="CBAM").cuda()
      elif self.args.network_type=="Pretrained_ResNet50_CBAM":
        gen_net=networks.Pretrained_ResNet50_CBAM().cuda()
      elif self.args.network_type=="Scikit_Learn_SVM":
        gen_net=networks.Scikit_Learn_SVM()
      else:
        Print("You didn't specify correct network type")
      
      # ================================================================================
      if self.args.network_type=="Scikit_Learn_SVM":
        return gen_net
      else:
        pass

      # ================================================================================
      # @ Configure multiple GPUs

      if self.args.use_multi_gpu=="True":
        num_gpu=torch.cuda.device_count()
        print("num_gpu",num_gpu)
        DEVICE_IDS=list(range(num_gpu))
        gen_encoder=nn.DataParallel(gen_encoder,device_ids=DEVICE_IDS)
      else: # args.use_multi_gpu=="False":
        pass

      # ================================================================================
      # @ Configure optimizer and scheduler

      optimizer=torch.optim.Adam(gen_net.parameters(),lr=self.lr)

      # ================================================================================
      # @ Load model for continuous training or test step

      if self.args.task_mode=="validation" or \
         self.args.task_mode=="submission" or \
         self.args.use_saved_model_for_continuous_train=="True":

        checkpoint_gener_direct_rgb=torch.load(
          self.args.model_save_dir+self.args.model_file_name_when_saving_and_loading_model)

        # # Get epoch info
        # start_epoch=checkpoint_gener_direct_rgb['epoch'] 

        # Apply loaded model to model
        gen_net.load_state_dict(checkpoint_gener_direct_rgb['state_dict']) 

        # Apply loaded model to optimizer
        optimizer.load_state_dict(checkpoint_gener_direct_rgb['optimizer'])

      else: # use_saved_model_for_continuous_train is False
        pass

      # ================================================================================
      # @ Print network infomation
      gen_net_param=self.print_network(gen_net)
      print("gen_net:",gen_net_param)

      # ================================================================================
      return gen_net,optimizer

    else: # use_integrated_decoders=="False":
      pass
  
  def print_network(self,gen_net,struct=False):

    # If you want show the structure of network
    if struct==True:
      print(gen_net)

    # ================================================================================
    # Initialize number of parameters to 0  
    num_params=0

    # Iterate all parameters of net
    for param in gen_net.parameters():
      # Increment number of parameter
      num_params+=param.numel()

    return num_params
  
  def save_checkpoint(self,state,filename):
    torch.save(state,filename)

  def denorm(self,x):
    out=(x+1)/2
    return out.clamp(0,1)
  
  def remove_existing_gradients_before_starting_new_training(self):
    self.gen_net.zero_grad()

  def save_model_after_epoch(self,num_info):
    if self.args.use_integrated_decoders=="True":

      # Create directory if it doesn't exist
      if not os.path.exists(self.args.model_save_dir):
        os.makedirs(self.args.model_save_dir)

      # c model_name: name of model
      model_name=self.args.model_file_name_when_saving_and_loading_model.split(".")[0]

      # c net_path: path of model
      net_path=self.args.model_save_dir+model_name+"_"+num_info+".pth"

      # save model
      self.save_checkpoint(
        state={
          # 'epoch':one_ep+1,
          'state_dict':self.gen_net.state_dict(),
          'optimizer':self.optimizer.state_dict()},
        filename=net_path)

    else: # use_integrated_decoders is False
      pass

  def empty_cache_of_gpu_after_training_batch(self):
    # c num_gpus: num of GPUs
    num_gpus=torch.cuda.device_count()

    # Iterate all GPUs
    for gpu_id in range(num_gpus):
      # Set GPU
      torch.cuda.set_device(gpu_id)
      # Empty GPU
      torch.cuda.empty_cache()
