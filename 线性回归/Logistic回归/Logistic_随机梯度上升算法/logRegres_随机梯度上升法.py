#__author__: dongj
#date: 2018/7/2
'''
梯度上升法在每次更新回归系数时都要遍历整个数据集，该方法在处理100
个左右的数据集还尚可，太多数据集的话，计算复杂度太大，随机梯度上升法
一次只用一个样本点来更新回归系数。是一个在线学习方法。
'''
import numpy as np
def loadDataSet():
    dataMat=[];labelMat=[]
    fr=open('testSet.txt')
    for line in fr:
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))
def stocGradAscent(dataMatrix,classLabels):
    m,n=np.shape(dataMatrix)
    alpha=0.01
    weights=np.ones(n)
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights))
        error=classLabels[i]-h
        dataMatrix=np.array(dataMatrix)#由于dataMatrix是一个列表所以应该将变成跟其他一样的ndarray型
        weights=weights+alpha*error*dataMatrix[i]
    return weights