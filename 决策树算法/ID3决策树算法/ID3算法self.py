# -*- coding: utf-8 -*-
# @Time    : 2018/10/4 16:09
# @Author  : 老湖
# @FileName: ID3算法self.py
# @Software: PyCharm
# @qq    ：326883847
from math  import log
'''
功能：计算数据集的信息熵
输入为：数据集(mxn),其中最后一列标签值
输出为：信息熵
'''
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

'''
功能：划分数据集
输入：待划分的数据集,划分数据集的属性,需要返回的特征的值
输出：输出为数据集中属性为需要返回的特征且去掉了该属性的数据集
'''
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reduceFeatVec=featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1,:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

'''
功能：选择最好的数据集划分方式
输入：数据集
输出：最好划分数据集的属性
'''
def chooseBestFeatureToSplit(dataSet):
    numFeature=len(dataSet[0])-1
    BestEntropy = 0.0;BestFeature = -1;splitInfo=0.0
    baseInfo=calcShannonEnt(dataSet)
    for i in numFeature:#依次遍历所有的特征
        featureList=[example[i] for example in dataSet]
        featureSet=list(set(featureList))
        for feat in featureSet:
            subDataMat=splitDataSet(dataSet,i,feat)
            prob=len(subDataMat)/float(len(dataSet))
            splitInfo+=prob*calcShannonEnt(subDataMat)
        currtEntropy=baseInfo-splitInfo
        if currtEntropy>BestEntropy:
            BestInfo=BestEntropy
            BestFeature=i
    return BestFeature

'''
功能：确定划分后的某一属性的类标签(选择类标签中的最多的那个为该叶子节点的标签)
输入：叶子节点的类标签集合
输出：返回次数最多的分类名称
'''
import operator
def majorityCnt(classList):
    classLabel={}
    for classL in classList:
        if classL not in classLabel.keys():classLabel[classL]=0
        classLabel[classL]+=1
        SortclassLabel=sorted(classLabel.items,key=operator.itemgetter[1],reverse=True)
    return SortclassLabel[0][0]

'''
功能：创建树
输入：数据集,标签列表包含了数据集中所有特征的标签,通俗的说就是列名
输出：一棵树
'''
def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel=labels[bestFeat]
    myTree={bestFeatureLabel:{}}
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    featValuesLabels=list(set(featValues))
    for feat in featValuesLabels:
        sublabels=labels[:]
        myTree[bestFeatureLabel][feat]=createTree(splitDataSet(dataSet,bestFeat,feat),sublabels)
    return myTree


'''
绘制图形
功能：获取叶子节点的数目和树的层数
输入：一个树
输出：叶子节点的数目和树的层数
'''
def getNumLeafs(myTree):
    numLeafs=0
    firstStr=myTree.keys()[0]#最佳划分属性点
    secondDict=myTree[firstStr]#最佳划分属性后的键值，包含的是一个字典，key为属性值，value为标签
    for key in secondDict.key():
        if type(secondDict[key]).__name__=='dict':
            numLeafs+=getNumLeafs(secondDict[key])
        else:
            numLeafs+=1
    return numLeafs