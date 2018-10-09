# -*- coding: utf-8 -*-
# @Time    : 2018/8/28 19:13
# @Author  : 老湖
# @FileName: 线性判别分析LDA.py
# @Software: PyCharm
# @qq    ：326883847
import numpy as np
def LoadData(FileName):#watermelon_3a.txt 
    DataArr=[];LableArr=[]
    with open(FileName) as fr:
        fileArr=[]
        for line in fr.readlines():
            fileArr.append(line.strip().split(' '))
    fileArr=np.array(fileArr,dtype='float')
    DataArr=fileArr[:,1:-1]
    LabelArr=fileArr[:,-1]
    return DataArr,LabelArr


def LDA_class(DataArr,LabelArr,n_class):
    m,n=np.shape(DataArr)
    LabelSet=list(set(LabelArr))
    DataSet={}
    #print('标签的类别有如下几种:',LabelSet)
    for i in LabelSet:
        DataSet[i]=DataArr[LabelArr==int(i)]
    Sw=np.zeros((n,n))
    Sb=np.zeros((n,n))
    dtMeanAll=[]
    for dt in DataSet.values():
        #print(dt,'展示数据')
        dtMean=np.mean(dt,axis=0,dtype='float')
        dtMeanAll.append(dtMean)
        Sw=np.add(Sw,np.dot((dt-dtMean).T,(dt-dtMean)))
        #print(Sw,'展示数据Sw')
        #print(np.shape(Sw), '展示数据Sw')
        u = np.mean(DataArr, axis=0)
        #print(u,'展示数据u')
        #print(np.mean(dt,axis=0),'展示数据均值')
        #print(Sb, '展示Sb')
        #print(np.shape(Sb),'展示Sb的维度')
        Sb =np.add(Sb,np.shape(dt)[0]*np.dot((np.mean(dt,axis=0)-u).T,np.mean(dt,axis=0)-u))
        #print(Sb,'展示Sb的值')
    if n==2:
        U, sigma, V = np.linalg.svd(Sw)
        Sw_inv=V.T*np.linalg.inv(np.diag(sigma))* U.T
        #print(np.linalg.inv(Sw), '展示Sw_inv的值')
        return np.dot(Sw_inv,((dtMeanAll[0]-dtMeanAll[1]).reshape(n,1)))
    else:
        from heapq import nlargest
        a,b=np.linalg.eig(Sw.I*Sb)
        a_index=a.index(nlargest(n_class,a))
        w=b[:,a_index]
        #print(w,'权值的大小')
    return w










