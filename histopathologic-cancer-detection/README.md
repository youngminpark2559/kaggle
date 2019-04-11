
#### Libraries
- Python 3.6
- PyTorch 1.0.1.post2
- CUDA V9.0.176
- CuDNN v7.4.2.24
- And others which you can install whenever you run into unmet-dependencies



! pip install albumentations
#! pip install pretrainedmodels
! pip install pytorchcv


#### Used techniques
- Python shell scripting to split excessively large dataset into separated directory
Then, you will easily upload split files onto Google colaboratory as well as you will easily access each directory to see the image files
- Train on Google colaboratory with Google drive storing entire pathology dataset
- Grad CAM to see where neural network pays attention to by analyzing gradient values of last conv layer  
You can check my summaries and comments if you want to briefly see it  
https://youngminpark2559.github.io/ml_cv_p/Grad-CAM_Visual_Explanations_from_Deep_Networks_via_Gradient-based_Localization/summaries_and_notes.html
- CBAM attention modules which help the neural network to see better place (where) and better things (what) for target class  
You can check my summaries and comments if you want to briefly see it  
https://youngminpark2559.github.io/ml_cv_p/CBAM_Convolutional_Block_Attention_Module/paper_summary.html
- Metrics to inspect dataset before training and to evaluate performance of the deep learning model after training
Accuracy, Precision, Recall, ROC Curve, F1 Score, etc  
If you want to briefly see the concept of metric, you can check this out  
1. For metrics for regression (L1 norm, L2 norm, etc), For metrics for classification (accuracy, precision, recall)  
https://youngminpark2559.github.io/mltheory/terry/YouTube/001_005_Metrics_for_deep_learrning_classification_Accuracy_Precision_Recall.html
2. Supplementary for accuracy, precision, recall. And for ROC curve, AUC  
https://youngminpark2559.github.io/mltheory/terry/YouTube/001_006_ROC_curve_AUC_Precision_Recall.html

#### Train information
- Epoch: 
- Batch size: 100 (processed by K80 11GB GPU memory of Google colaboratory notebook)
- Weight on loss value: 10.0 (final_loss_value=10.0*loss_value)
- Train dataset split: train 90%, validation 10%
- Input image size: center (48,48,3) from original (96,96,3)
- Tested network
1. ResNet50 (Pretrained, Finetune last FC layer)
2. Custom CNN+FC
3. ResNet50+CBAM attention module which is originated from
https://github.com/Jongchan/attention-module

#### Work flow on this project that I went through
1. Run util to process big data  
/prj_root/src/utils/utils_split_files_into_directories_to_easily_upload_files_onto_colab.py
2. Upload big data onto Google drive
3. Edit argument to whatever you want
/home/young/Kaggle_histopathologic-cancer-detection/my_mode_new_architecture/prj_root/src/argument_api/argument_api_class.py
3. After full train and you download saved model files
4. Load saved model files and make prediction over test dataset on you local PC to make submission to Kaggle

#### Result
- Grad CAM output  
![alt text](https://raw.githubusercontent.com/youngminpark2559/kaggle/master/histopathologic-cancer-detection/src/utils_analyzing_result/Grad_CAM_output/00a718935d8faf4795a0d1576f6fffd636bfe4ef_ori.png)  
![alt text](https://raw.githubusercontent.com/youngminpark2559/kaggle/master/histopathologic-cancer-detection/src/utils_analyzing_result/Grad_CAM_output/00a718935d8faf4795a0d1576f6fffd636bfe4ef.png)  
![alt text](https://raw.githubusercontent.com/youngminpark2559/kaggle/master/histopathologic-cancer-detection/src/utils_analyzing_result/Grad_CAM_output/01cfd9afa45f8bf05012bff600504fc8549b9b9c_ori.png)  
![alt text](https://raw.githubusercontent.com/youngminpark2559/kaggle/master/histopathologic-cancer-detection/src/utils_analyzing_result/Grad_CAM_output/01cfd9afa45f8bf05012bff600504fc8549b9b9c.png)  
![alt text](https://raw.githubusercontent.com/youngminpark2559/kaggle/master/histopathologic-cancer-detection/src/utils_analyzing_result/Grad_CAM_output/02dcc96b7dfd481e806d86d3344a433340f9ec21_ori.png)  
![alt text](https://raw.githubusercontent.com/youngminpark2559/kaggle/master/histopathologic-cancer-detection/src/utils_analyzing_result/Grad_CAM_output/02dcc96b7dfd481e806d86d3344a433340f9ec21.png)  

#### Opinion on trainng process and result
- 

#### Backpropagation, autograd, gradient in PyTorch
- If you want to read above topics which are relevant to Grad CAM which deals with gradient values, check this out
https://youngminpark2559.github.io/prac_ml/pytorch/kykim/002_autograd_and_Variable.html





Input size should be large. feature extraction.

Not. figure. but should kepp in mind. train with partial dataset.


Pretrain not good.


learning rate start 0.001
update 5 times.

latency too high, local 6GB gpu over 15 times (literally extremly and incomparable) faster
for example, start training 0 batch 10 batch 
