# -*- coding: utf-8 -*-
# @Time    : 2018/8/31 20:38
# @Author  : 老湖
# @FileName: bin.py
# @Software: PyCharm
# @qq    ：326883847
import numpy as np
from 使用多项式模型实现的朴素贝叶斯算法过滤网站的恶意留言 import loadDataSet,createVocabList,setOfWords2Vec,trainNB0,classifyNB
from 使用朴素贝叶斯过滤垃圾邮件 import spamTest
from 使用朴素贝叶斯分类器从个人广告中获取区域倾向 import localwords,getTopWords
import feedparser
#加载训练集数据
dataArr,LabelArr=loadDataSet()

#创建训练集词向量
dataArrVec=createVocabList(dataArr)

#显示每个评论的在词向量中的出现表示
y_pre=setOfWords2Vec(dataArrVec,dataArr[3])
#print(y_pre,'显示每个评论的在词向量中的出现表示')

#求属于两个类别的词量数和最后的概率
trainMat=[]
for postinDoc in dataArr:
    trainMat.append(setOfWords2Vec(dataArrVec,postinDoc))
p0V,p1V,pAb=trainNB0(trainMat,LabelArr)#由训练集得到每一类中的数据统计
# print(p0V,'显示属于第零类的词量总数')
# print(p1V,'显示属于第一类的词向量总数')
# print(pAb,'显示概率')

#测试一个数据样本属于哪一类
print('-------------------------测试一个数据样本属于哪一类----------------------')
testEntry=['love','my','dalmation']
thisDoc=np.array(setOfWords2Vec(dataArrVec,testEntry))
print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

#测试实例2
print('--------------------测试实例2---------------------')
testEntry=['stupid','garbage']
thisDoc=np.array(setOfWords2Vec(dataArrVec,testEntry))
print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))

#使用朴素贝叶斯过滤垃圾邮件
print('------------------------使用朴素贝叶斯过滤垃圾邮件------------------------')
spamTest()


#使用朴素贝叶斯分类器从个人广告中获取区域倾向
print('------------------------使用朴素贝叶斯分类器从个人广告中获取区域倾向------------------------')
ny=feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
sf=feedparser.parse('http://rss.tom.com/happy/happy.xml')
# print(ny,'显示输入')
# print(sf,'显示数据')
vocabList,pSF,pNY=localwords(ny,sf)
getTopWords(ny,sf)
