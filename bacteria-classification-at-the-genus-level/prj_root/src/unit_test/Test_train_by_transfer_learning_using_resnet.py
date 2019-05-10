# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/my_model/prj_root/src/unit_test && \
# rm e.l && python Test_train_by_transfer_learning_using_resnet.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import unittest
import sys,os,copy,argparse
sys.path.insert(0,'/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/my_model/prj_root')

# ================================================================================
from src.utils import utils_create_argument as utils_create_argument

from src.train import train_by_transfer_learning_using_resnet as train_by_transfer_learning_using_resnet

# ================================================================================
class KnownValues(unittest.TestCase):

  def test_utils_net_for_cgintrinsic_net(self):
    # ================================================================================
    # Arrange
    args=utils_create_argument.return_argument()

    # ================================================================================
    # Act
    netG=train_by_transfer_learning_using_resnet.train(args)
    # print('netG',type(netG))
    # <class 'prj_root.utils.utils_net_for_cgintrinsic_net.MultiUnetGenerator'>

    # ================================================================================
    # Assert
    # self.assertEqual(3,netG)
    self.assertIsInstance(netG,utils_net_for_cgintrinsic_net.MultiUnetGenerator)

  # ================================================================================

if __name__=='__main__':

  # --------------------------------------------------------------------------------

  unittest.main()