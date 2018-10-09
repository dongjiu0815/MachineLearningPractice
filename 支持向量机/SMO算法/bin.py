# -*- coding: utf-8 -*-
# @Time    : 2018/9/6 20:14
# @Author  : 老湖
# @FileName: bin.py
# @Software: PyCharm
# @qq    ：326883847

import svmMLiA
from Simple_smo import smoSimple
import numpy as np
from PlattSMO import smoP,calcWs,smoPK,testRbf
#加载数据
dataArr,labelArr=svmMLiA.loadDataSet('testSet.txt')
# print(labelArr,'显示标签数据')
# print(dataArr,'显示x数据')

#实现简化版的SMO算法
print('--------------显示简化版SMO算法--------------')
b,alphas=smoSimple(dataArr,labelArr,0.6,0.001,40)
print(alphas[alphas>0],'显示alpha中那些不是零的数')
print(np.shape(alphas[alphas>0]),'显示支持向量的个数')
for i in range(100):
    if alphas[i]>0.0:
        print(dataArr[i],labelArr[i])

#实现完整的有核SMO算法
print('-------------实现完整的有核SMO算法----------------')
b,alphas=smoPK(dataArr,labelArr,0.6,0.001,40)
#print(alphas,'显示alphas的值')
#print(b,'显示b的值')
wsK=calcWs(alphas,dataArr,labelArr)
print(wsK,'ws显示权重')

#实现完整版的无核SMO算法
print('-------------实现完整的无核SMO算法----------------')
b,alphas=smoP(dataArr,labelArr,0.6,0.001,40)
#print(alphas,'显示alphas的值')
#print(b,'显示b的值')
ws=calcWs(alphas,dataArr,labelArr)
print(ws,'wsk显示权重')

#利用和函数进行分类的径向基测试函数
print('------------------利用和函数进行分类的径向基测试函数--------------------')
testRbf()
