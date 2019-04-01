from types import SimpleNamespace

# ================================================================================
def return_argument():
  args=SimpleNamespace(
    batch_size='11',
    check_input_output_via_multi_gpus='False',
    epoch='2',
    input_size='256',
    leaping_batchsize_for_saving_model='1',
    measure_train_time='True',
    model_save_dir='./ckpt',
    scheduler='None',
    seed=42,
    text_file_for_paths_dir='/mnt/1T-5e7/image/whole_dataset',
    train_method='train_by_transfer_learning_using_resnet',
    train_mode='True',
    use_augmentor='True',
    use_integrated_decoders='True',
    use_local_piecewise_constant_loss='True',
    use_loss_display='True',
    use_multi_gpu='False',
    use_saved_model='False')
  return args