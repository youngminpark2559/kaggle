from skimage.transform import resize
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)

def load_csv_in_pd(path):
    data=pd.read_csv(path)
    return data

def prepare_dataset(pd_csv_f,test=False):
    # Generate train dataset
    if test==False:
        np_arr=np.array(pd_csv_f)
        # print("np_arr",np_arr.shape)
        # np_arr (42000, 785)

        # c train_y: labels of train dataset
        train_y=np_arr[:,0]
        # print("train_y",train_y)
        # train_y [1 0 1 ... 7 6 9]

        # c img: train images
        img=np_arr[:,1:]
        # print("img",img.shape)
        # img (42000, 784)
        
        # c w_img: width length of image
        w_img=np.sqrt(img.shape[1]).astype("uint8")
        # print("w_img",w_img)
        # w_img 28
        
        # c train_X: reshaped image as X of train dataset
        train_X=img.reshape((-1,w_img,w_img))
        # print("train_X",train_X.shape)
        # train_X (42000, 28, 28)

        return train_X,train_y
    else:
        np_arr=np.array(pd_csv_f)
        # print("np_arr",np_arr.shape)
        # np_arr (28000, 784)

        # c w_img: width length of image
        w_img=np.sqrt(np_arr.shape[1]).astype("uint8")
        # print("w_img",w_img)
        # w_img 28
        
        # c train_X: reshaped image as X of test dataset
        test_X=np_arr.reshape((-1,w_img,w_img))
        # print("test_X",test_X.shape)
        # test_X (28000, 28, 28)

        # print("test_X",test_X)

        return test_X

def split_dataset(train_X,train_y):
    train_X,val_X,train_y,val_y=train_test_split(
        train_X,train_y,test_size=0.3,random_state=42)
    return train_X,val_X,train_y,val_y

def reshape_data_X(train_X,test_X,val_X=None):
    if val_X!=None:
        rs_train_X=train_X.reshape(train_X.shape[0],-1)
        rs_test_X=test_X.reshape(test_X.shape[0],-1)
        rs_val_X=val_X.reshape(val_X.shape[0],-1)
        return rs_train_X,rs_test_X,rs_val_X
    else:
        rs_train_X=train_X.reshape(train_X.shape[0],-1)
        rs_test_X=test_X.reshape(test_X.shape[0],-1)
        return rs_train_X,rs_test_X

def create_ph_arr(shape,data_type):
    if data_type=="uint8":
        ph_arr=np.zeros(shape,dtype=np.uint8)
        return ph_arr
    elif data_type=="bool":
        ph_arr=np.zeros(shape,dtype=np.bool)
        return ph_arr

def rs_tr_img(train_ids,path_train,X_train,Y_train):
    for n, id_ in enumerate(train_ids):
        # print(n)
        # print(id_)
        # 0
        # 9e9ce2e1ce.png

        path=path_train
        img=Image.open(path+'/images/'+id_)
        img=np.array(img)
        # print("img",img)
        # [[[114 114 114]
        #   [136 136 136]
        #   [158 158 158]
        #   ...
        #   [103 103 103]
        #   [101 101 101]
        #   [105 105 105]]
        
        #  [[120 120 120]
        #   [136 136 136]
        #   [151 151 151]
        
        x=img[:,:,1]
        x=resize(x,(128,128,1),mode='constant',preserve_range=True)
        # print("x",x.shape)
        # x (128, 128, 1)
        # plt.imshow(x.squeeze(),cmap="gray")
        # plt.show()

        X_train[n]=x
        
        mask=Image.open(path+'/masks/'+id_)
        # print("np.array(mask)",np.array(mask).shape)
        # np.array(mask) (101, 101)
        mask=(np.array(mask)[:,:]).astype("bool")
        
        Y_train[n]=resize(mask,(128,128,1),mode='constant',preserve_range=True)
    
    return X_train,Y_train

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)