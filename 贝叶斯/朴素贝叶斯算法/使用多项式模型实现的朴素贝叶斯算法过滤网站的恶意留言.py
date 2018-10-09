# -*- coding: utf-8 -*-
# @Time    : 2018/9/13 11:08
# @Author  : 老湖
# @FileName: 基于多项式模型实现的朴素贝叶斯算法.py
# @Software: PyCharm
# @qq    ：326883847
import numpy as np
def loadDataSet():
    postingList=[['my','dog','has','flea','problems','help','please'],\
                 ['maybe','not','take','him','to','dog','park','stupid'],\
                 ['my','dalmation','is','so','cute','I','love','him'],\
                 ['stop','posting','stupid','worthless','garbage'],\
                 ['mr','licks','ate','my','steak','how','to','stop','him'],\
                 ['quit','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1]
    return postingList,classVec
def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet|set(document)
    vocabSet=sorted(vocabSet,key=str.lower)
    print(vocabSet)
    return list(vocabSet)
#setOfWords2Vec编写的是词集模型，他将没个词是否出现作为一个特征
def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:print('the word:%s is not in my Vocabulary!'% word)
    return returnVec
#bagOfWords2VecMN编写的是词袋模型，它将每个单词出现的总次数作为特征
def bagOfWord2VecMN(vocabLIst,inputSet):
    returnVec=[0]*len(vocabLIst)
    for word in inputSet:
        if word in vocabLIst:
            returnVec[vocabLIst.index(word)]+=1
        else:print('the word:%s is not in my Vocabulary!' % word)
    return returnVec
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    #print(numTrainDocs,'xians')#结果为样本数
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    # p0Num=np.zeros(numWords);p1Num=np.zeros(numWords)
    # p0Denom=0.0;p1Denom=0.0
    #为了防止出现某个条件该概率为零使得最后的结果为零上述的代码改为下面的
    p0Num = np.ones(numWords);
    p1Num = np.ones(numWords)
    p0Denom=2.0;p1Denom=2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    # p1Vect=p1Num/p1Denom     #change to log()
    # p0Vect=p0Num/p0Denom     #change to log()
    #为了防止最后计算出现下溢的现象上诉的代码做如下改变
    p1Vect = np.log(p1Num / p1Denom)  # change to log()
    p0Vect=np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive
def classifyNB(vec2Classify,p0Vec,p1Vecl,pClass1):
    p1=sum(vec2Classify*p1Vecl)+np.log(pClass1)
    p0=sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

