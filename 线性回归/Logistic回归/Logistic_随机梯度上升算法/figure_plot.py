#__author__: dongj
#date: 2018/7/2
from logRegres_随机梯度上升法 import loadDataSet
import numpy as np
import matplotlib.pyplot as plt
def bestplotFit(weights):
    datamat,labelmat=loadDataSet()
    datamat=np.array(datamat)
    m=np.shape(datamat)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(m):
        if int(labelmat[i])==1:
            xcord1.append(datamat[i,1]);ycord1.append(datamat[i,2])
        else:
            xcord2.append(datamat[i,1]);ycord2.append(datamat[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=np.arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()