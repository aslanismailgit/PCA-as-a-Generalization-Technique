#%%
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
#%%
path2PCA = "./PCAdrop/"
for r in ["99.txt","95.txt","85.txt","80.txt","75.txt","65.txt","60.txt",
            "55.txt","50.txt","45.txt","40.txt","35.txt","30.txt"]:
    for i,f in enumerate(os.listdir(path2PCA)):

        if r in f :
            df = pd.read_csv(os.path.join(path2PCA, f),
                            names=["numOfComps", "loss", "acc"], 
                            header=None)
            print(f)
            m_index = df["numOfComps"].idxmin()
            print ("min value","\n",df["numOfComps"][m_index-2:m_index+3].values )
            print (m_index,"\n" )

#%%
path2PCA = "./PCAdrop/"
dr_rates = ["99","95","85","8_","75","65","6_",
            "55","5_","45","4_","35","3_"]
for r in dr_rates:
    s = r.split(".")[0]
    plt.figure()
    for i,f in enumerate(os.listdir(path2PCA)):
        
        if r in f and "txt" in f :
            df = pd.read_csv(os.path.join(path2PCA, f),
                            names=["numOfComps", "loss", "acc"], 
                            header=None)
            
            
            plt.plot(df["numOfComps"][:100],".")
    plt.title(f'Var keeped : %{s} -- \nFirst 100 batch trainings of 10 runs - 50 epochs')
    plt.ylabel("Number of Components")
    plt.xlabel("Number of Batch Trainngs")
    pltname = "./findings/" + "NumOfComps_" + str(s) + ".jpg"
    print(pltname)
    # plt.savefig(pltname, dpi=380)
    plt.show()
#%%
