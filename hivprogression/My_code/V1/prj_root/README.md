
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
1.. Correlation between factors<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V_0002/prj_root/img_out/Analyze_train_data/2019_05_02_10%3A49%3A06.png" alt="drawing" width="300" height="300"/><br/>
..(1) Meaning: there are nagative and positive relationships between factors.<br/>
..(2) Code: /src/utils_preanalyze_data/utils_preanalyze_data_module.py<br/>
          visualize_images(args)<br/>

2.. Frequence distribution of label data (Resp)<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V_0002/prj_root/img_out/Analyze_train_data/2019_05_02_11%3A59%3A41.png" alt="drawing" width="300" height="300"/><br/>
..(1) Meaning:<br/>
....1) There are imbalanced labels which should be solved for accurate training the model<br/>

3.. Frequence distribution of train data (PR Seq)<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V_0002/prj_root/img_out/Analyze_train_data/2019_05_02_13%3A27%3A45.png" alt="drawing" width="300" height="300"/><br/>
..(1) Meaning:<br/>
....1) All data is unique<br/>

4.. Frequence distribution of train data (VL-t0)<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V_0002/prj_root/img_out/Analyze_train_data/2019_05_02_12%3A35%3A28.png" alt="drawing" width="300" height="300"/><br/>
..(1) Meaning:<br/>
....1) Overall, frequent distribution looks Gaussian normal distribution except for periodic low values like 1 and 2<br/>

5.. Frequence distribution of train data (CD4-t0)<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V_0002/prj_root/img_out/Analyze_train_data/2019_05_02_13%3A50%3A59.png" alt="drawing" width="300" height="300"/><br/>
..(1) Meaning:<br/>
....1) Data is biased to the the left region (small values)<br/>

#### Analyze label data
1. Normalize CD4 data from [0,1200] scale to [0,1] scale<br/>
Distribution doesn't change.<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V_0002/prj_root/img_out/Process_data/2019_05_02_20%3A56%3A06.png" alt="drawing" width="300" height="300"/><br/>

#### Train result  
1.. Support Vector Machine classifier (Scikit-Learn)<br/>

================================================================================
@.. 1-Fold evaluation

```
1 Binary confusion matrix
[[249   0]
 [ 39  19]]

2 report

                        precision    recall  f1-score   support

class Non tumor (neg)       0.86      1.00      0.93       249
    class Tumor (pos)       1.00      0.33      0.49        58

            micro avg       0.87      0.87      0.87       307
            macro avg       0.93      0.66      0.71       307
         weighted avg       0.89      0.87      0.85       307

3 accuracy_score 0.8729641693811075
4 precision_score 1.0
5 recall_score 0.3275862068965517
6 f1_score 0.49350649350649356

================================================================================
@ 2-Fold evaluation

1 Binary confusion matrix
[[243   0]
 [ 42  21]]

2 report
                       precision    recall  f1-score   support

class Non tumor (neg)       0.85      1.00      0.92       243
    class Tumor (pos)       1.00      0.33      0.50        63

            micro avg       0.86      0.86      0.86       306
            macro avg       0.93      0.67      0.71       306
         weighted avg       0.88      0.86      0.83       306

3 accuracy_score 0.8627450980392157
4 precision_score 1.0
5 recall_score 0.3333333333333333
6 f1_score 0.5

================================================================================
@ 3-Fold evaluation

1 Binary confusion matrix 
[[241   0]
 [ 65   0]]

2 report
                       precision    recall  f1-score   support

class Non tumor (neg)       0.79      1.00      0.88       241
    class Tumor (pos)       0.00      0.00      0.00        65

            micro avg       0.79      0.79      0.79       306
            macro avg       0.39      0.50      0.44       306
         weighted avg       0.62      0.79      0.69       306

3 accuracy_score 0.7875816993464052
4 precision_score 0.0
5 recall_score 0.0
6 f1_score 0.0
```
7 ROC curve<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V1/prj_root/img_out/Analyze_result/2019_05_04_07%3A19%3A33.png" alt="drawing" width="300" height="200"/><br/> <img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V1/prj_root/img_out/Analyze_result/2019_05_04_07%3A19%3A46.png" alt="drawing" width="200" height="200"/><br/> <img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V1/prj_root/img_out/Analyze_result/2019_05_04_07%3A19%3A56.png" alt="drawing" width="200" height="200"/><br/>
