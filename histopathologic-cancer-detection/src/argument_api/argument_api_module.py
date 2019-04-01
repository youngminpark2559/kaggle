import datetime
import sys,os,copy,argparse

class Argument_API_class(argparse.ArgumentParser):
  def __init__(self):
    super(Argument_API_class,self).__init__()
    self.add_argument("--train_mode",default=True)
    self.add_argument("--train_method",help="train_by_custom_net,train_by_transfer_learning_using_resnet")
    self.add_argument("--use_saved_model",default=False)
    self.add_argument("--model_save_dir")
    self.add_argument("--model_file_name_when_saving_and_loading_model")
    self.add_argument("--epoch")
    self.add_argument("--batch_size")
    self.add_argument("--use_augmentor",default=False)
    self.add_argument("--use_integrated_decoders",default=True)
    self.add_argument("--use_loss_display",default=True)
    self.add_argument("--input_size")
    self.add_argument("--text_file_for_paths_dir")
    self.add_argument("--use_multi_gpu",default=False)
    self.add_argument("--check_input_output_via_multi_gpus",default=False)
    self.add_argument("--measure_train_time",default=False)
    self.add_argument("--scheduler",default=None)
    self.add_argument('--seed',type=int,default=42,metavar='S',help='random seed (default: 42)')
    self.args=self.parse_args()
