
#### Introduction
- Problem: predict and classify 0 or 1 by training predicting model using HIV related data
- Competition page: https://www.kaggle.com/c/hivprogression/  
- There are 28 proteins in labels  
- Multiple proteins can be shown in one image  
- 28 proteins have different shapes
- Multiple labels means one image can have multiple labels like [one_image,[protein_label1,protein_label2,protein_label3]]  
- And dataset is quite imbalance which means, for example, protein_label1 shows up very frequently, protein_label2 shows very rarely  
- You should manage that imbalance in the process of training

#### Libraries
- Python 3.6
- PyTorch 1.0.1.post2
- CUDA V10.0.130
- CuDNN v7.4
- And others which you can install whenever you run into unmet-dependencies

#### Used techniques
- `CBAM attention modules` which help the neural network to see better place (where) and better things (what) for target class  
https://github.com/youngminpark2559/kaggle/blob/master/histopathologic-cancer-detection/src/networks/networks.py#L864  
https://github.com/youngminpark2559/kaggle/blob/master/histopathologic-cancer-detection/src/networks/cbam.py

You can check my summaries and comments if you want to briefly see it  
https://youngminpark2559.github.io/ml_cv_p/CBAM_Convolutional_Block_Attention_Module/paper_summary.html  

- `K-Fold Cross Train and Validation`  
(1) Devide paths into 3 folds (3 train folds, 3 validation folds)  
https://github.com/youngminpark2559/kaggle/blob/master/human-protein-atlas-image-classification/src/train/train_by_transfer_learning_using_resnet.py#L79
(2) Loop over the 3 folds  
https://github.com/youngminpark2559/kaggle/blob/master/human-protein-atlas-image-classification/src/train/train_by_transfer_learning_using_resnet.py#L109

- `Focal Loss`  
(1) Code  
https://github.com/youngminpark2559/kaggle/blob/master/human-protein-atlas-image-classification/src/loss_functions/loss_functions_module.py#L4

- `Metrics`  
(1) F1 scores for multi-label & multi-class problem  
https://github.com/youngminpark2559/kaggle/blob/master/human-protein-atlas-image-classification/src/metrics/metrics_module.py#L2

#### Visualize train data  
1.. Correlation between factors  
<img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V1/prj_root/img_out/Analyze_train_data/2019_05_02_10%3A49%3A06.png" alt="drawing" width="300" height="300"/>  

(1).. Meaning: there are nagative and positive relationships between factors.  
(2).. Code: /src/utils_preanalyze_data/utils_preanalyze_data_module.py  
visualize_images(args)  

2.. Frequence distribution of label data (Resp)  
<img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V1/prj_root/img_out/Analyze_train_data/2019_05_02_11%3A59%3A41.png" alt="drawing" width="300" height="300"/>  
(1).. Meaning:  
1).. There are imbalanced labels which should be solved for accurate training the model  

3.. Frequence distribution of train data (PR Seq)<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V1/prj_root/img_out/Analyze_train_data/2019_05_02_13:27:45.png" alt="drawing" width="300" height="300"/><br/>
(1).. Meaning:<br/>
1).. All data is unique<br/>


3.. Frequence distribution of train data (cd4-t0)  
<img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V1/prj_root/img_out/Analyze_train_data/2019_05_02_12%3A35%3A28.png" alt="drawing" width="300" height="300"/>  
(1).. Meaning:  
1).. Overall, frequent distribution looks Gaussian normal distribution  
except for periodic low values like 1 and 2  

4.. Frequence distribution of train data (VL-t0)  
<img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V1/prj_root/img_out/Analyze_train_data/2019_05_02_13%3A00%3A42.png" alt="drawing" width="300" height="300"/>  
(1).. Meaning:  
1).. All data is unique  

















#### Analyze label data
(1) Frequent distribution of train label data  
<img src="./img_out/Analyze_label_data/train_label_distribution.png" alt="drawing" width="600" height="300"/>  
1) Meaning: label data is very imbalance; Nucleoplasm very often shows, Rods & rings shows very rarely.  
2) Code: /src/utils_preanalyze_data/utils_preanalyze_data_module.py

(2) Frequent distribution of number of labels to each image
<img src="./img_out/Analyze_label_data/Frequent_distribution_of_number_of_labels_to_each_img.png" alt="drawing" width="600" height="300"/>  
1) Meaning: Many images have 1 label  
2) Code: /src/utils_preanalyze_data/utils_preanalyze_data_module.py  

(3) Correlation of proteins  
<img src="./img_out/Analyze_label_data/correlation_of_proteins.png" alt="drawing" width="600" height="300"/>  
1) Meaning: No much of correlation of couple of proteins  
Some proteins has positive correlation (blue) (if one proteins shows more, corresponding other proten also shows more)  
Some proteins has negative correlation (red) (if one proteins shows more, corresponding other proten also shows less)  
2) Code: /src/utils_preanalyze_data/utils_preanalyze_data_module.py  

#### Train workflow  
```
3fold_train_img_path,3folds_train_label,3folds,validation_path,3folds_validation_label=\
   get_3_folds_of_train_and_validation_path(entire_path)

for one_k in range(3):
  for one_epoch in range(all_epochs):
    for one_batch in range(all_batch):
      prediction=network(input)
      loss=focal_loss(prediction,label)
      loss.backward()
      optimizer.step()

  for vali_idx,validation_data in enumerate(validation_dataloader):
    with torch.no_grad():
      prediction=network(input)
      f1_score_value=f1_score(prediction,label)
```


#### Train result  
(1) Decrease of loss value  
1) Visualization  
<img src="./img_out/Result_scores/loss.png" alt="drawing" width="600" height="300"/>  

(2) F1 scores  
1) Visualization  
<img src="./img_out/Result_scores/f1_score.png" alt="drawing" width="600" height="300"/>  

#### Backpropagation, autograd, gradient in PyTorch
- If you want to read above topics which are relevant to Grad CAM which deals with gradient values, check this out  
https://youngminpark2559.github.io/prac_ml/pytorch/kykim/002_autograd_and_Variable.html
