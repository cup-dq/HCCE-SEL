import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.utils import resample
from resampled import resampled
def RKNN(X_train,y_train,n_nbors):
    X_resampled,y_resampled=resampled(X_train,y_train)
    neigh = NearestNeighbors(n_neighbors=n_nbors)
    neigh.fit(X_resampled)
    #print(neigh.kneighbors()[1])
    arr=neigh.kneighbors()[1]
    #判断多数类的K近邻有无少数类样本，即查询少数类的反K近邻
    #前t为为多数类
    l=[]
    ss = pd.to_numeric(y_train.iloc[:, 0])
    for i, element in enumerate(ss):
        if element == 0:
            l.append(i)
    t=len(l)
    # print("多数类个数")
    # print(t)
    arr=arr[0:t]
    #print(arr)
    l1=[]
    i=0
    for x in arr:
        i=i+1
        for y in x:
            if y>=t-1:
                l1.append(i-1)

    list1 = list(set(l1))
    return(list1)