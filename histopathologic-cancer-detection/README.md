
#### Libraries
- Python 3.6
- PyTorch 1.0.
- CUDA
- CuDNN
- And others which you can install whenever you run into unmet-dependencies

#### Used techniques
- Python shell scripting to split excessively large dataset into separated directory
Then, you will easily upload split files onto Google colaboratory as well as you will easily access each directory to see the image files
- Train on Google colaborabory
- Grad CAM ([You can check my summaries and comments if you want to briefly see it](https://youngminpark2559.github.io/ml_cv_p/Grad-CAM_Visual_Explanations_from_Deep_Networks_via_Gradient-based_Localization/summaries_and_notes.html))
- CBAM ([You can check my summaries and comments if you want to briefly see it](https://youngminpark2559.github.io/ml_cv_p/CBAM_Convolutional_Block_Attention_Module/paper_summary.html))
- Metrics

#### Train information
- Epoch
- Train dataset split: train 90%, validation 10%
- Input image size (96,96,3)
- Tested network
1. ResNet50
2. Custom CNN+FC
3. ResNet50+CBAM

#### Work flow on this project that I went through
1. Run util to process big data  
/prj_root/src/utils/utils_split_files_into_directories_to_easily_upload_files_onto_colab.py
2. Upload big data onto colab
3. Edit argument to whatever you want
/home/young/Kaggle_histopathologic-cancer-detection/my_mode_new_architecture/prj_root/src/argument_api/argument_api_class.py
3. After full train and you download saved model files
4. Load saved model files and make prediction over test dataset on you local PC to make submission to Kaggle

#### Opinion on trainng process and result
- 

#### PyTorch backprop, autograd, gradient post
- If you want to read them, related to Grad CAM which deals with gradient values,
check [this](https://youngminpark2559.github.io/prac_ml/pytorch/kykim/002_autograd_and_Variable.html) out

