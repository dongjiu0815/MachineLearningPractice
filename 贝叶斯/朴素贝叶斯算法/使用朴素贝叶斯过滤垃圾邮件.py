# -*- coding: utf-8 -*-
# @Time    : 2018/9/14 10:02
# @Author  : 老湖
# @FileName: 使用朴素贝叶斯过滤垃圾邮件.py
# @Software: PyCharm
# @qq    ：326883847
import numpy as np
def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet|set(document)
    vocabSet=sorted(vocabSet,key=str.lower)
    #print(vocabSet,'创建词向量集合')
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:print('the word:%s is not in my Vocabulary!'% word)
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    #print(numTrainDocs,'xians')#结果为样本数
    # print(type(trainMatrix),'显示数据类型')
    # print(np.shape(trainMatrix),'显示数据的维度')
    #print(trainMatrix,'显示数据')
    numWords=len(trainMatrix[0])
    #print(numWords,'显示数据的维度')
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

#文本解析
def textParse(bigString):
    import re
    listOfTokens=re.split(r'[^\w*]',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>1]

def spamTest():
    docList,classList,fullText=[],[],[]
    for i in range(1,26):
        wordlist=textParse(open('spam/%d.txt'%i).read())
        docList.append(wordlist)
        fullText.extend(docList)
        classList.append(1)
        wordlist=textParse(open('ham/%d.txt'%i).read())
        docList.append(wordlist)
        fullText.extend(docList)
        classList.append(0)
    print(docList,'显示创建之前的数据集合')
    vocabList=createVocabList(docList)
    trainingSet=list(range(50));testSet=[]
    for i in range(10):
        randIndex=int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[];trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount=0
    for docIndex in testSet:
        wordVector=setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print('the error rate is: ',float(errorCount)/len(testSet))


