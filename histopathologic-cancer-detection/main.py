# conda activate py36gputorch100 && \
# cd /home/young/Kaggle_histopathologic-cancer-detection/my_mode_new_architecture/prj_root

# Train
# rm e.l && python main.py \
# --train_mode=True \
# --batch_size=3 \
# --epoch=10 \
# --use_saved_model=False \
# --model_save_dir="./ckpt" \
# --model_file_name_when_saving_and_loading_model="/tumor_pretrained_resnet_fifty.pth" \
# --use_augmentor=True \
# --use_integrated_decoders=True \
# --use_loss_display=False \
# --input_size=48 \
# --use_multi_gpu=False \
# --text_file_for_paths_dir="/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data" \
# --check_input_output_via_multi_gpus=False \
# --measure_train_time=True \
# --scheduler=None \
# --train_method="train_by_transfer_learning_using_resnet" \
# 2>&1 | tee -a e.l && code e.l

# Test
# rm e.l && python main.py \
# --train_mode=False \
# --batch_size=11 \
# --epoch=2 \
# --use_saved_model=True \
# --model_save_dir="./ckpt" \
# --model_file_name_when_saving_and_loading_model="/tumor_pretrained_resnet_fifty.pth" \
# --use_augmentor=True \
# --use_integrated_decoders=True \
# --use_loss_display=False \
# --input_size=48 \
# --use_multi_gpu=False \
# --text_file_for_paths_dir="/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data" \
# --check_input_output_via_multi_gpus=False \
# --measure_train_time=True \
# --scheduler=None \
# --train_method="train_by_transfer_learning_using_resnet" \
# 2>&1 | tee -a e.l && code e.l

# @ Basic modules
import datetime
import sys,os,copy,argparse
# print('current working directory',os.getcwd())
# print('path of current file',os.path.abspath(__file__))
# print('dir name of current file',os.path.dirname(os.path.abspath(__file__)))
# @ PyTorch modules
import torch
# @ src/train
from src.train import train_by_transfer_learning_using_resnet as train_by_transfer_learning_using_resnet
# @ src/argument_api
from src.argument_api import argument_api_module as argument_api_module

def start_func(args):
  if args.train_method=="train_by_custom_net":
    train_script.train(args)
  elif args.train_method=="train_by_transfer_learning_using_resnet":
    train_by_transfer_learning_using_resnet.train(args)
  else:
    pass

if __name__=="__main__":
  # c argument_api: instance of Argument_API_class
  argument_api=argument_api_module.Argument_API_class()
  # print("argument_api",argument_api)
  # Argument_API_class(
  #   prog='main.py',
  #   usage=None,
  #   description=None,
  #   formatter_class=<class 'argparse.HelpFormatter'>,
  #   conflict_handler='error',
  #   add_help=True)

  # c args: member attribute from argument_api
  args=argument_api.args
  # print("args",args)
  # Namespace(
  #   batch_size='3',
  #   check_input_output_via_multi_gpus='False',
  #   epoch='10',
  #   input_size='48',
  #   measure_train_time='True',
  #   model_file_name_when_saving_and_loading_model='/tumor_pretrained_resnet_fifty.pth',
  #   model_save_dir='./ckpt',
  #   scheduler='None',
  #   seed=42,
  #   text_file_for_paths_dir='/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data',
  #   train_method='train_by_transfer_learning_using_resnet',
  #   train_mode='True',
  #   use_augmentor='True',
  #   use_integrated_decoders='True',
  #   use_loss_display='False',
  #   use_multi_gpu='False',
  #   use_saved_model='False')
  # start function
  start_func(args)
