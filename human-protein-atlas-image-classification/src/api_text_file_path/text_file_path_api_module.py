import sys,os,copy,argparse

class Path_Of_Text_Files():
  def __init__(self,args):
    # dir_where_text_file_for_image_paths_is_in
    # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data

    # ================================================================================
    # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/Path_of_train_images.txt

    self.image_data=args.dir_where_text_file_for_image_paths_is_in+"/Path_of_train_images.txt"

    # ================================================================================
    # /mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train.csv

    self.label_data=args.dir_where_text_file_for_image_paths_is_in+"/train.csv"
