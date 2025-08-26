import os
import numpy as np
import pandas as pd


def convert(file_a,file_b):
    coor_tissue=pd.read_csv(file_a)#"/GTEX-1A8G6-2926.svs.csv"
    slide_features=np.load(file_b)#"GTEX-1A8G6-2926_features.npy"
    a=slide_features
    p=a.shape[0]
    q=a.shape[1]
    t=a.shape[2]
    c=np.empty((p,q,t),dtype=np.float16) #
    c[:] = np.NaN
    print(c.shape)
    print(a.shape)

    coor=coor_tissue.iloc[:,1:]
    print(coor)
    for i in range(len(np.array(coor))): 
        x=coor.iloc[i,0]
        y=coor.iloc[i,1]
        c[:,y,x]=a[:,y,x]
    print(file_a)
    name=file_a.split("/")[-1].split(".")[0]
    np.save("/no-bounding-features-remove-bg/"+str(name+'_features.npy'), c)
    return None


import glob
csv_folder="/nerve-tibial-coor"
csv_path_list = glob.glob(csv_folder+ '/*.csv')
csv_path_list.sort()
npy_folder="no-bounding-features/"
npy_path_list = glob.glob(npy_folder+ '/*.npy')
npy_path_list.sort()


for i in range(len(csv_path_list)):
    file_a=csv_path_list[i]
    file_b=npy_path_list[i]
    file_a_ID=file_a.split("/")[-1].split(".svs")[0]
    file_b_ID=file_b.split("/")[-1].split("_features")[0]
    if file_a_ID==file_b_ID:
        print("true")
        print(file_a)
        print(file_b)
        convert(file_a,file_b)
    else:
        print(file_a_ID)
        print(file_b_ID)
