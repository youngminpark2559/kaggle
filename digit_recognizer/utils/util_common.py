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
