# conda activate py36gputorch041
# cd /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/utils/
# rm e.l && python utils_image.py 2>&1 | tee -a e.l && code e.l

# ================================================================================
from types import SimpleNamespace

# ================================================================================
def return_argument():
  args=SimpleNamespace(
    batch_size='22',
    check_input_output_via_multi_gpus='False',
    epoch='3',
    input_size='64',
    leaping_batchsize_for_saving_model='1',
    measure_train_time='True',
    model_save_dir='./ckpt',
    seed=42,
    dir_where_text_file_for_image_paths_is_in='/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data',
    train_method='train_by_transfer_learning_using_resnet',
    task_mode='True',
    use_augmentor='True',
    use_integrated_decoders='True',
    use_local_piecewise_constant_loss='True',
    use_loss_display='True',
    use_multi_gpu='False',
    use_saved_model_for_continuous_train='False')
  return args
