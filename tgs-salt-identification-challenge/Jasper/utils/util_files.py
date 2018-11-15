import sys,os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)

def load_csv_in_pd(path):
    data=pd.read_csv(path)
    return data

def create_res_csv(predict_test,test_X,solver):
    test_X_df=pd.DataFrame(
        test_X.reshape(test_X.shape[0],test_X.shape[1]*test_X.shape[2]))
    # (28000,784)

    test_X_df['label']=predict_test

    cols=test_X_df.columns.tolist()
    cols.insert(0, cols.pop(-1))

    test_X_df=test_X_df[cols]
    # print("test_X_df",test_X_df.shape)
    # test_X_df (28000, 785)

    # c w_te_img: width of test images
    w_te_img=int(np.sqrt(test_X_df.shape[1]-1))
    # print("w_te_img",w_te_img)
    # w_te_img 28

    # c te_labels: labels of test images
    te_labels=np.array(test_X_df)[:,0].astype("uint8")
    # c te_imgs: images of test images
    te_imgs=np.array(test_X_df)[:,1:].reshape(-1,w_te_img,w_te_img)

    # c o_p_t_img: one prediced test image
    # for o_p_t_img in range(50):
    #     plt.imshow(te_imgs[o_p_t_img,:,:],cmap="gray")
    #     plt.title(te_labels[o_p_t_img])
    #     plt.show()

    s_sub=load_csv_in_pd("./all/sample_submission_original.csv")
    # print('s_sub.loc[:,"Label"]',s_sub.loc[:,"Label"].shape)
    # s_sub.loc[:,"Label"] (28000,)
    
    s_sub.loc[:,"Label"]=te_labels
    s_sub.to_csv("./all/submission_"+str(solver)+".csv", sep=',',index=False)

    s_sub=load_csv_in_pd("./all/submission_"+str(solver)+".csv")
    # print("s_sub",s_sub.head())

def get_data_ids(path):
    ids=next(os.walk(path+"/images"))[2]
    return ids
