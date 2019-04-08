import sys,os,copy,argparse

class Path_Of_Text_Files():
  def __init__(self,args):
    # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data/trn_txt_file_processed_small.txt
    # self.tumor_trn=args.dir_where_text_file_for_image_paths_is_in+"/trn_txt_file_processed_small.txt"
    self.tumor_trn=args.dir_where_text_file_for_image_paths_is_in+"/trn_txt_file_processed.txt"
    # self.tumor_trn=args.dir_where_text_file_for_image_paths_is_in+"/trn_txt_file_processed_colab.txt"

    # self.tumor_lbl=args.dir_where_text_file_for_image_paths_is_in+"/train_labels_small.csv"
    self.tumor_lbl=args.dir_where_text_file_for_image_paths_is_in+"/train_labels.csv"
