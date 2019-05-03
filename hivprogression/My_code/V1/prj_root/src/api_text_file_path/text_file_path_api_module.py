import sys,os,copy,argparse

class Path_Of_Text_Files():
  def __init__(self,args):
    # dir_where_text_file_for_image_paths_is_in
    # /mnt/1T-5e7/mycodehtml/prac_data_science/kaggle/hivprogression/My_code/Data

    # ================================================================================
    # /mnt/1T-5e7/mycodehtml/prac_data_science/kaggle/hivprogression/My_code/Data/train_csv_path.txt

    self.train_data=args.dir_where_text_file_for_image_paths_is_in+"/train_csv_path.txt"

    # ================================================================================
    # /mnt/1T-5e7/mycodehtml/prac_data_science/kaggle/hivprogression/My_code/Data/test_csv_path.txt

    self.test_data=args.dir_where_text_file_for_image_paths_is_in+"/test_csv_path.txt"
