# -*- coding: utf-8 -*-
# @Time    : 2018/9/14 16:57
# @Author  : 老湖
# @FileName: 使用朴素贝叶斯分类器从个人广告中获取区域倾向.py
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
def bagOfWord2VecMN(vocabLIst,inputSet):
    returnVec=[0]*len(vocabLIst)
    for word in inputSet:
        if word in vocabLIst:
            returnVec[vocabLIst.index(word)]+=1
def textParse(bigString):
    import re
    listOfTokens=re.split(r'[^\w*]',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>1]
def calcMostFreq(vocabList,fullText):
    """
       Function：   计算出现频率

       Args：       vocabList：词汇表
                   fullText：全部词汇

       Returns：    sortedFreq[:30]：出现频率最高的30个词
       """
    import operator
    freqDict={}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    #operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为一些序号（即需要获取的数据在对象中的序号
    sortedFreq=sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]
def localwords(feed1,feed0):
    """
        Function：   RSS源分类器

        Args：       feed1：RSS源
                    feed0：RSS源

        Returns：    vocabList：词汇表
                    p0V：类别概率向量
                    p1V：类别概率向量
        """
    import feedparser
    # 初始化数据列表
    docList = [];classList = [];fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    print(feed1['entries'],'显示数据')
    # 导入文本文件
    for i in range(minLen):
        # 切分文本
        wordList = textParse(feed1['entries'][i]['summary'])
        # 切分后的文本以原始列表形式加入文档列表
        docList.append(wordList)
        # 切分后的文本直接合并到词汇列表
        fullText.extend(wordList)
        # 标签列表更新
        classList.append(1)
        # 切分文本
        wordList = textParse(feed0['entries'][i]['summary'])
        # 切分后的文本以原始列表形式加入文档列表
        docList.append(wordList)
        # 切分后的文本直接合并到词汇列表
        fullText.extend(wordList)
        # 标签列表更新
        classList.append(0)
    # 获得词汇表
    vocabList = createVocabList(docList)
    # 获得30个频率最高的词汇
    top30Words = calcMostFreq(vocabList, fullText)
    # 去掉出现次数最高的那些词
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = list(range(2 * minLen));testSet = []
    # 随机构建测试集，随机选取二十个样本作为测试样本，并从训练样本中剔除
    for i in range(20):
        # 随机得到Index
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        # 将该样本加入测试集中
        testSet.append(trainingSet[randIndex])
        # 同时将该样本从训练集中剔除
        del (trainingSet[randIndex])
    # 初始化训练集数据列表和标签列表
    trainMat = [];trainClasses = []
    # 遍历训练集
    for docIndex in trainingSet:
        # 词表转换到向量，并加入到训练数据列表中
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        # 相应的标签也加入训练标签列表中
        trainClasses.append(classList[docIndex])
    # 朴素贝叶斯分类器训练函数
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    # 初始化错误计数
    errorCount = 0
    # 遍历测试集进行测试
    for docIndex in testSet:
        # 词表转换到向量
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        # 判断分类结果与原标签是否一致
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            # 如果不一致则错误计数加1
            errorCount += 1
            # 并且输出出错的文档
            print("classification error", docList[docIndex])
    # 打印输出信息
    print('the erroe rate is: ', float(errorCount) / len(testSet))
    # 返回词汇表和两个类别概率向量
    return vocabList, p0V, p1V
def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localwords(ny,sf)
    topNY=[];topSF=[]
    for i in range(len(p0V)):
        if p0V[i]>-6.0:topSF.append((vocabList[i],p0V[i]))
        if p1V[i]>-6.0:topNY.append((vocabList[i],p1V[i]))
        sortedSF=sorted(topSF,key=lambda pair:pair[1],reverse=True)
        print('SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF')
        for item in sortedSF:
            print(item[0])
        sortedNY=sorted(topNY,key=lambda pair:pair[1],reverse=True)
        print('NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY')
        for item in sortedNY:
            print(item[0])

