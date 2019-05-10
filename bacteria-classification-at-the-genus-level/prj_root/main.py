# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/My_code/V_0001/prj_root

# ================================================================================
# Train
# rm e.l && python main.py \
# --start_mode="train" \
# --task_mode="train" \
# --train_method="CNN" \
# --use_saved_model_for_continuous_train=False \
# --model_save_dir="./ckpt" \
# --model_file_name_when_saving_and_loading_model="/tumor_pretrained_resnet50_cbam.pth" \
# --epoch=9 \
# --batch_size=2 \
# --use_augmentor=True \
# --use_integrated_decoders=True \
# --use_loss_display=True \
# --dir_where_text_file_for_image_paths_is_in="/mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level" \
# --use_multi_gpu=False \
# --check_input_output_via_multi_gpus=False \
# --measure_train_time=True \
# --optimizer=None \
# --network_type="ResNet50_CBAM" \
# --leapping_term_when_displaying_loss=1000 \
# --leapping_term_when_saving_model_after_batch=1000 \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
# dir_where_text_file_for_image_paths_is_in="./../Data"

# --start_mode="preanalyze_data" \
# --start_mode="train" \

# --network_type="ResNet50_CBAM" \
# --network_type="Scikit_Learn_SVM" \

# --train_method="CNN" \
# --train_method="Scikit_Learn_SVM" \
# --train_method="xgboost" \
# --train_method="xgboost_one_hot" \

# ================================================================================
# Validataion
# rm e.l && python main.py \
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

# ================================================================================
# Submission
# rm e.l && python main.py \
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
from src.train import train_by_Scikit_Learn_SVM as train_by_Scikit_Learn_SVM
from src.train import train_by_xgboost as train_by_xgboost
from src.train import train_by_xgboost_resample as train_by_xgboost_resample
from src.train import train_by_CNN as train_by_CNN

from src.api_argument import argument_api_module as argument_api_module

from src.utils_preanalyze_data import utils_preanalyze_data_module as utils_preanalyze_data_module

# ================================================================================
def start_train(args):
  if args.train_method=="train_by_custom_net":
    train_script.train(args)
  elif args.train_method=="train_by_transfer_learning_using_resnet":
    train_by_transfer_learning_using_resnet.train(args)
  elif args.train_method=="Scikit_Learn_SVM":
    train_by_Scikit_Learn_SVM.train(args)
  elif args.train_method=="xgboost":
    train_by_xgboost.train(args)
  elif args.train_method=="xgboost_resample":
    train_by_xgboost_resample.train(args)
  elif args.train_method=="CNN":
    train_by_CNN.train(args)
  else:
    pass

# ================================================================================
def preanalyze_data(args):
  utils_preanalyze_data_module.analyze_data(args)
  # utils_preanalyze_data_module.see_correlation_of_data(args)
  # utils_preanalyze_data_module.distribution_of_label_in_trn_data(args)
  # utils_preanalyze_data_module.distribution_of_PR_Seq_in_trn_data(args)
  # utils_preanalyze_data_module.distribution_of_VL_t0_in_trn_data(args)
  # utils_preanalyze_data_module.distribution_of_CD4_t0_in_trn_data(args)

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

  if args.start_mode=="preanalyze_data":
    preanalyze_data(args)
  elif args.start_mode=="train":
    start_train(args)
