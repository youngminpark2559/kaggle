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
        train_X=img.reshape((-1,w_img,w_img))/255.0
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
        test_X=np_arr.reshape((-1,w_img,w_img))/255.0
        # print("test_X",test_X.shape)
        # test_X (28000, 28, 28)

        # print("test_X",test_X)

        return test_X

