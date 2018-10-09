# -*- coding: utf-8 -*-
# @Time    : 2018/8/10 17:20
# @Author  : 老湖
# @FileName: 改进你的随机梯度下降算法求解线性规划.py
# @Software: PyCharm
# @qq    ：326883847
def loadDataSet(filename):
    import numpy as np
    dataArr,labelArr,cur=[],[],[]
    with open(filename) as textfile:
        for line in textfile.readlines():
            cur.append(line.strip().split('\t'))
    cur=np.array(cur,dtype='float32')
    cur=np.random.shuffle(cur)
    dataArr,labelArr=cur[:,:-1],cur[:,-1]
    return dataArr,labelArr
def standRegRes_ImproveSGD(dataMatrix,classLabels,numiter=150):
    import numpy as np
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    count=0.0
    for j in range(numiter):
        for i in range(m):
            count+=1
            alpha =4.0/(i+j+1.0)+0.01#第一处改进
            randomindex=int(np.random.uniform(0,m))#第二处改进
            # print(weights.shape)
            # print(type(weights))
            # print(dataMatrix.shape)
            # print(type(dataMatrix))
            h=sum(dataMatrix[randomindex]*weights)
            error=classLabels[randomindex]-h
            dataMatrix=np.array(dataMatrix)#由于dataMatrix是一个列表所以应该将变成跟其他一样的ndarray型
            weights=weights+alpha*error*dataMatrix[randomindex]
            np.delete(dataMatrix,randomindex,0)
    return weights,count