#__author__: dongj
#date: 2018/7/1
import numpy as np
def loadDataSet():
    dataMat=[];labelMat=[]
    fr=open('testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat
def sigmod(inX):
    return 1.0/(1+np.exp(-inX))
def gradAscent(dataMatIn,classLabels):
    dataMatrix=np.mat(dataMatIn)
    labelMat=np.mat(classLabels).transpose()
    m,n=np.shape(dataMatrix)
    alpha=0.001
    maxCycles=500
    weights=np.ones((n,1))
    for k in range(maxCycles):
        h=sigmod(dataMatrix*weights)
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights

# test your trained Logistic Regression model given test set
def testLogRegres(weights, test_x, test_y):
    numSamples, numFeatures = np.shape(test_x)
    matchCount = 0
    print(numFeatures,numSamples)
    for i in range(numSamples):
        print(test_x[i],'显示维度')
        print(weights,'xianshi ')
        predict = sigmod(np.dot(np.array(test_x[i]).reshape(1,-1) , weights.reshape(-1,1)))[0, 0] > 0.5
        if predict == bool(test_y[i]):
            matchCount += 1
    accuracy = float(matchCount) / numSamples
    return accuracy