#%%
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

#%%
path2base = "./runResults/basemodel"
path2drop = "./runResults/dropout/"
path2PCA = "./runResults/PCAdrop/"

colnames = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
colnames = ['loss', 'val_loss']
# colnames = ['val_accuracy']
trs = 0.995
for metr in colnames:
    plt.figure()
#--------------------------BASE-----------------------------------    
   
    temp = np.empty((50,10))
    path_temp = path2base
    dirname = path_temp.split("/")[-1]
    # print(dirname)
    for i, f in enumerate(os.listdir(path2base)):
        # print(i,f)
        df = pd.read_csv(os.path.join(path2base, f))
        df.drop(df.columns[0],axis=1, inplace=True)
        temp[:,i] = df[metr]
    base_m = np.mean(temp,axis=1)
    plt.plot(np.arange(20, 50),base_m[20:],  ".", label=dirname)    
    

    for j, dirname in enumerate(os.listdir(path2drop)):
        path_temp = os.path.join(path2drop, dirname)
        # print(j,dirname, path_temp)
        temp = np.empty((50,10))
        for i, f in enumerate(os.listdir(path_temp)):
            # print(i,f)
            df = pd.read_csv(os.path.join(path_temp, f))
            df.drop(df.columns[0],axis=1, inplace=True)
            temp[:,i] = df[metr]
        m = np.mean(temp,axis=1)
        if np.mean(m[40:])< trs * np.mean(base_m[40:]):    
            plt.plot(np.arange(20, 50),m[20:],label=dirname)         

    for j, dirname in enumerate(os.listdir(path2PCA)):
        path_temp = os.path.join(path2PCA, dirname)
        # print(j,dirname, path_temp)
        if "var" not in path_temp:
            temp = np.empty((50,10))
            for i, f in enumerate(os.listdir(path_temp)):
                # print(i,f)
                df = pd.read_csv(os.path.join(path_temp, f))
                df.drop(df.columns[0],axis=1, inplace=True)
                temp[:,i] = df[metr]
            m = np.mean(temp,axis=1)
            if np.mean(m[40:])< trs * np.mean(base_m[40:]):    
                plt.plot(np.arange(20, 50),m[20:], "--", label=dirname)   
    # plt.xticks()
    plt.title(metr.capitalize())
    plt.legend()
    pltname = "./findings/" + metr.capitalize() + "_20_50_.jpg"
    discr = "_mean_below_" + str (trs) 
    pltname = "./findings/" + metr.capitalize() + discr + "_.jpg"

    print(pltname)
    plt.savefig(pltname, dpi=360)
    plt.show()
#%%
