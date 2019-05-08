
#### Introduction
- Page: https://www.kaggle.com/c/hivprogression/
- Problem: predict (or classify) 0 or 1 against given HIV related feature data
- To do that, you need to train classifier by given train dataset

================================================================================
#### Explanation on data
- Column-1: Patient's ID
- Column-2: Response status to the treatment ("1": improved (got better), "0": otherwise)
- Column-3: DNA sequence to create protease
- Column-4: DNA sequence to create reverse transciptase
- Column-5: "viral load severity" at the beginning of therapy (log-10 units)
- Column-6: CD4 (cluster structures on the surface of immune cells such as T helper cells, monocytes, macrophages) count at the beginning of therapy

================================================================================
#### Libraries
- Python 3.6
- PyTorch 1.0.1.post2
- CUDA V10.0.130
- CuDNN v7.4
- Scikit-Learn
- And others which you can install whenever you run into unmet-dependencies

================================================================================
#### Used techniques
- `K-Fold Cross Train and Validation`  
(1) Devide paths into 3 folds (3 train folds, 3 validation folds)  
https://github.com/youngminpark2559/kaggle/blob/master/human-protein-atlas-image-classification/src/train/train_by_transfer_learning_using_resnet.py#L79
(2) Loop over the 3 folds  
https://github.com/youngminpark2559/kaggle/blob/master/human-protein-atlas-image-classification/src/train/train_by_transfer_learning_using_resnet.py#L109

- `Metrics`  
(1) F1 scores for multi-label & multi-class problem  
https://github.com/youngminpark2559/kaggle/blob/master/human-protein-atlas-image-classification/src/metrics/metrics_module.py#L2

================================================================================
#### Visualize train data  
1.. Correlation between factors<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V1/prj_root/img_out/Analyze_train_data/2019_05_02_10%3A49%3A06.png" alt="drawing" width="300" height="300"/><br/>
..(1) Meaning: there are nagative and positive relationships between factors.<br/>
..(2) Code: /src/utils_preanalyze_data/utils_preanalyze_data_module.py<br/>
          visualize_images(args)<br/>

2.. Frequence distribution of label data (Resp)<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V1/prj_root/img_out/Analyze_train_data/2019_05_02_11%3A59%3A41.png" alt="drawing" width="300" height="300"/><br/>
..(1) Meaning:<br/>
....1) There are imbalanced labels which should be solved for accurate training the model<br/>

3.. Frequence distribution of train data (PR Seq)<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V1/prj_root/img_out/Analyze_train_data/2019_05_02_13%3A27%3A45.png" alt="drawing" width="300" height="300"/><br/>
..(1) Meaning:<br/>
....1) All data is unique<br/>

4.. Frequence distribution of train data (VL-t0)<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V1/prj_root/img_out/Analyze_train_data/2019_05_02_12%3A35%3A28.png" alt="drawing" width="300" height="300"/><br/>
..(1) Meaning:<br/>
....1) Overall, frequent distribution looks Gaussian normal distribution except for periodic low values like 1 and 2<br/>

5.. Frequence distribution of train data (CD4-t0)<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V1/prj_root/img_out/Analyze_train_data/2019_05_02_13%3A50%3A59.png" alt="drawing" width="300" height="300"/><br/>
..(1) Meaning:<br/>
....1) Data is biased to the the left region (small values)<br/>

#### Analyze label data
1.. Normalize CD4 data from [0,1200] scale to [0,1] scale<br/>
Distribution doesn't change.<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V1/prj_root/img_out/Process_data/2019_05_02_20%3A56%3A06.png" alt="drawing" width="300" height="300"/><br/>

================================================================================
#### Train result  
1.. Support Vector Machine classifier (Scikit-Learn)<br/>

```
@ 1-Fold evaluation

Confusion matrix
[[249   0]
 [ 39  19]]

Report

                        precision    recall  f1-score   support

class Non tumor (neg)       0.86      1.00      0.93       249
    class Tumor (pos)       1.00      0.33      0.49        58

            micro avg       0.87      0.87      0.87       307
            macro avg       0.93      0.66      0.71       307
         weighted avg       0.89      0.87      0.85       307

Accuracy_score 0.8729641693811075
Precision_score 1.0
Recall_score 0.3275862068965517
F1_score 0.49350649350649356

================================================================================
@ 2-Fold evaluation

Confusion matrix 
[[243   0]
 [ 42  21]]

Report
                       precision    recall  f1-score   support

class Non tumor (neg)       0.85      1.00      0.92       243
    class Tumor (pos)       1.00      0.33      0.50        63

            micro avg       0.86      0.86      0.86       306
            macro avg       0.93      0.67      0.71       306
         weighted avg       0.88      0.86      0.83       306

Accuracy_score 0.8627450980392157
Precision_score 1.0
Recall_score 0.3333333333333333
F1_score 0.5

================================================================================
@ 3-Fold evaluation

Confusion matrix 
[[241   0]
 [ 65   0]]

Report
                       precision    recall  f1-score   support

class Non tumor (neg)       0.79      1.00      0.88       241
    class Tumor (pos)       0.00      0.00      0.00        65

            micro avg       0.79      0.79      0.79       306
            macro avg       0.39      0.50      0.44       306
         weighted avg       0.62      0.79      0.69       306

Accuracy_score 0.7875816993464052
Precision_score 0.0
Recall_score 0.0
F1_score 0.0
```

ROC curve<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V1/prj_root/img_out/Analyze_result/2019_05_04_07%3A19%3A33.png" alt="drawing" width="200" height="200"/> <img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V1/prj_root/img_out/Analyze_result/2019_05_04_07%3A19%3A46.png" alt="drawing" width="200" height="200"/> <img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V1/prj_root/img_out/Analyze_result/2019_05_04_07%3A19%3A56.png" alt="drawing" width="200" height="200"/><br/>

================================================================================<br/>
2.. XGBoost classifier<br/>

```
@ 1-Fold evaluation

Confusion matrix  
[[247   2]
 [  9  49]]

Report
                        precision    recall  f1-score   support
class Non tumor (neg)       0.96      0.99      0.98       249
    class Tumor (pos)       0.96      0.84      0.90        58

            micro avg       0.96      0.96      0.96       307
            macro avg       0.96      0.92      0.94       307
         weighted avg       0.96      0.96      0.96       307

Accuracy_score 0.9641693811074918
Precision_score 0.9607843137254902
Recall_score 0.8448275862068966
F1_score 0.8990825688073395

================================================================================
@ 2-Fold evaluation

Confusion matrix 
[[243   0]
 [ 11  52]]

Report
                        precision    recall  f1-score   support

class Non tumor (neg)       0.96      1.00      0.98       243
    class Tumor (pos)       1.00      0.83      0.90        63

            micro avg       0.96      0.96      0.96       306
            macro avg       0.98      0.91      0.94       306
         weighted avg       0.97      0.96      0.96       306

Accuracy_score 0.9640522875816994
Precision_score 1.0
Recall_score 0.8253968253968254
F1_score 0.9043478260869565

================================================================================
@ 3-Fold evaluation

Confusion matrix 
[[227  14]
 [ 44  21]]

Report
                        precision    recall  f1-score   support

class Non tumor (neg)       0.84      0.94      0.89       241
    class Tumor (pos)       0.60      0.32      0.42        65

            micro avg       0.81      0.81      0.81       306
            macro avg       0.72      0.63      0.65       306
         weighted avg       0.79      0.81      0.79       306

Accuracy_score 0.8104575163398693
Precision_score 0.6
Recall_score 0.3230769230769231
F1_score 0.42
```

ROC curve<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V1/prj_root/img_out/Analyze_result/2019_05_04_08%3A21%3A05.png" alt="drawing" width="200" height="200"/> <img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V1/prj_root/img_out/Analyze_result/2019_05_04_08%3A21%3A20.png" alt="drawing" width="200" height="200"/> <img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V1/prj_root/img_out/Analyze_result/2019_05_04_08%3A21%3A32.png" alt="drawing" width="200" height="200"/><br/>

================================================================================<br/>
@ Random re-up-sample for lack label data<br/>
```
Confusion matrix 
[[125  21]
 [ 14 134]]

Report
                           precision    recall  f1-score   support

    class Non tumor (neg)       0.90      0.86      0.88       146
        class Tumor (pos)       0.86      0.91      0.88       148

                micro avg       0.88      0.88      0.88       294
                macro avg       0.88      0.88      0.88       294
             weighted avg       0.88      0.88      0.88       294

Accuracy_score 0.8809523809523809
Precision_score 0.864516129032258
Recall_score 0.9054054054054054
F1_score 0.8844884488448843
```

ROC curve<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/kaggle/master/hivprogression/My_code/V1/prj_root/img_out/Analyze_result/2019_05_08_21%3A47%3A25.png" alt="drawing" width="200" height="200"/>
```

================================================================================<br/>
To do:<br/>

# [checkbox:unchecked] 1.. Normalize CD4 and VL data into [0,1] scale?<br/>
# [checkbox:checked] 2.. I performed K-fold train&evaluaion to resolve imbalance of label<br/>
This time I will perform upsample on smaller labeled data<br/>
