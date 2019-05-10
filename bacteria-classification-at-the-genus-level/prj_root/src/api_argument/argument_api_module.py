import datetime
import sys,os,copy,argparse

class Argument_API_class(argparse.ArgumentParser):
  def __init__(self):
    super(Argument_API_class,self).__init__()
    self.add_argument("--start_mode",help="preanalyze_data,train")
    self.add_argument("--task_mode",default=True)
    self.add_argument("--train_method",help="train_by_custom_net,train_by_transfer_learning_using_resnet")
    self.add_argument("--use_saved_model_for_continuous_train",default=False)
    self.add_argument("--model_save_dir")
    self.add_argument("--model_file_name_when_saving_and_loading_model")
    self.add_argument("--epoch")
    self.add_argument("--batch_size")
    self.add_argument("--use_augmentor",default=False)
    self.add_argument("--use_integrated_decoders",default=True)
    self.add_argument("--use_loss_display",default=True)
    self.add_argument("--dir_where_text_file_for_image_paths_is_in")
    self.add_argument("--use_multi_gpu",default=False)
    self.add_argument("--check_input_output_via_multi_gpus",default=False)
    self.add_argument("--measure_train_time",default=False)
    self.add_argument("--optimizer",default=None)
    self.add_argument("--network_type")
    self.add_argument("--leapping_term_when_displaying_loss")
    self.add_argument("--leapping_term_when_saving_model_after_batch")

    self.args=self.parse_args()
