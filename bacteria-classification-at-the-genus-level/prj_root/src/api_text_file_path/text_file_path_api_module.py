import sys,os,copy,argparse

class Path_Of_Text_Files():
  def __init__(self,args):

    # ================================================================================
    # /mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level/trn_imgs_paths.txt
    self.train_data=args.dir_where_text_file_for_image_paths_is_in+"/trn_imgs_paths.txt"

    # ================================================================================
    # /mnt/1T-5e7/mycodehtml/bio_health/Bacteria/bacteria-classification-at-the-genus-level/Data/bacteria-classification-at-the-genus-level/test_imgs_paths.txt
    self.label_data=args.dir_where_text_file_for_image_paths_is_in+"/trn_imgs_labels.csv"
