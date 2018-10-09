#__author__: dongj
#date: 2018/7/2
'''
第一处改进:alpha每次迭代的时候都会调整会缓解数据波动和高频波动，
虽会减少但不会减少到零，保证多次迭代后新数据仍然具有一定的影响，如果
处理动态变化可以加大上面的常数
第二处改进：随机选取样本来更新回归系数。减少周期性的波动
第三次改进：增加了迭代次数作为第三个参数默认为150
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
def stocGradAscentImprove(dataMatrix,classLabels,numiter=150):
    m,n=np.shape(dataMatrix)
    weights=np.ones(n)
    for j in range(numiter):
        for i in range(m):
            alpha =4.0/(i+j+1.0)+0.01#第一处改进
            randomindex=int(np.random.uniform(0,m))#第二处改进
            h=sigmoid(sum(dataMatrix[randomindex]*weights))
            error=classLabels[randomindex]-h
            dataMatrix=np.array(dataMatrix)#由于dataMatrix是一个列表所以应该将变成跟其他一样的ndarray型
            weights=weights+alpha*error*dataMatrix[randomindex]
            np.delete(dataMatrix,randomindex,0)
    return weights