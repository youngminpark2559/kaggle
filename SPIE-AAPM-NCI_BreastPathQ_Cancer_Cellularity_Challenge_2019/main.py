# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/bio_health/SPIE-AAPM-NCI_BreastPathQ_Cancer_Cellularity_Challenge_2019/My_code/prj_root

# ================================================================================
# Train
# rm e.l && python main.py \
# --start_func_mode="pre_visualize_data" \
# --task_mode="train" \
# --train_method="train_by_transfer_learning_using_resnet" \
# --use_saved_model_for_continuous_train=True \
# --model_save_dir="./ckpt" \
# --model_file_name_when_saving_and_loading_model="/tumor_pretrained_resnet50_cbam.pth" \
# --epoch=9 \
# --batch_size=30 \
# --use_augmentor=True \
# --use_integrated_decoders=True \
# --use_loss_display=True \
# --dir_where_text_file_for_image_paths_is_in="/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data" \
# --use_multi_gpu=False \
# --check_input_output_via_multi_gpus=False \
# --measure_train_time=True \
# --optimizer=None \
# --network_type="ResNet50_CBAM" \
# --leapping_term_when_displaying_loss=1000 \
# --leapping_term_when_saving_model_after_batch=1000 \
# 2>&1 | tee -a e.l && code e.l

# dir_where_text_file_for_image_paths_is_in="./../Data"

# Validataion
# rm e.l && python main.py \
# --start_func_mode="pre_visualize_data" \
# --task_mode="validation" \
# --train_method="train_by_transfer_learning_using_resnet" \
# --use_saved_model_for_continuous_train=False \
# --model_save_dir="./ckpt" \
# --model_file_name_when_saving_and_loading_model="/tumor_pretrained_resnet50_cbam.pth" \
# --epoch=5 \
# --batch_size=30 \
# --use_augmentor=True \
# --use_integrated_decoders=True \
# --use_loss_display=True \
# --dir_where_text_file_for_image_paths_is_in="/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data" \
# --use_multi_gpu=False \
# --check_input_output_via_multi_gpus=False \
# --measure_train_time=True \
# --optimizer=None \
# --network_type="ResNet50_CBAM" \
# --leapping_term_when_displaying_loss=10 \
# --leapping_term_when_saving_model_after_batch=10 \
# 2>&1 | tee -a e.l && code e.l

# Submission
# rm e.l && python main.py \
# --start_func_mode="pre_visualize_data" \
# --task_mode="submission" \
# --train_method="train_by_transfer_learning_using_resnet" \
# --use_saved_model_for_continuous_train=False \
# --model_save_dir="./ckpt" \
# --model_file_name_when_saving_and_loading_model="/tumor_pretrained_resnet50_cbam.pth" \
# --epoch=5 \
# --batch_size=200 \
# --use_augmentor=True \
# --use_integrated_decoders=True \
# --use_loss_display=True \
# --dir_where_text_file_for_image_paths_is_in="/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data" \
# --use_multi_gpu=False \
# --check_input_output_via_multi_gpus=False \
# --measure_train_time=True \
# --optimizer=None \
# --network_type="ResNet50_CBAM" \
# --leapping_term_when_displaying_loss=10 \
# --leapping_term_when_saving_model_after_batch=10 \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import datetime
import sys,os,copy,argparse

import torch

# ================================================================================
from src.train import train_by_transfer_learning_using_resnet as train_by_transfer_learning_using_resnet

from src.argument_api import argument_api_module as argument_api_module

from src.analyze_pre import visualize_data as visualize_data

# ================================================================================
def start_func(args):
  if args.start_func_mode=="pre_visualize_data":
    visualize_data.visualize(args)
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
  #   measure_train_time='True',
  #   model_file_name_when_saving_and_loading_model='/tumor_pretrained_resnet_fifty.pth',
  #   model_save_dir='./ckpt',
  #   network_type='ResNet50_CBAM',
  #   dir_where_text_file_for_image_paths_is_in='/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data',
  #   train_method='train_by_transfer_learning_using_resnet',
  #   task_mode='True',
  #   use_augmentor='True',
  #   use_integrated_decoders='True',
  #   use_loss_display='False',
  #   use_multi_gpu='False',
  #   use_saved_model_for_continuous_train='False')

  start_func(args)
