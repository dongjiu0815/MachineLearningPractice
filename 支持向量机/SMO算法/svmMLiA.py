# -*- coding: utf-8 -*-
# @Time    : 2018/9/11 14:19
# @Author  : 老湖
# @FileName: svmMLiA.py
# @Software: PyCharm
# @qq    ：326883847
def loadDataSet(fileName):
    dataMat=[];labelMat=[]
    with open(fileName) as fr:
        for line in fr.readlines():
            lineArr=line.strip().split('\t')
            dataMat.append([float(lineArr[0]),float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
    return dataMat,labelMat
def selectJrand(i,m):
    import numpy as np
    j=i
    while(j==i):
        j=int(np.random.uniform(0,m))
    return j
def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

