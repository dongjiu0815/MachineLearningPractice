# -*- coding: utf-8 -*-
# @Time    : 2018/10/4 16:08
# @Author  : 老湖
# @FileName: bin.py
# @Software: PyCharm
# @qq    ：326883847

from ID3算法 import CalcShannonEnt,createDataSet,chooseBestFeatureToSplit,createTree

#加载数据
'''
功能：生成数据集和标签集(列名)
输入：空
输出：数据集myDat(最后一列为标签值)，标签集(列名)
'''
myDat,labels=createDataSet()

#测试嫡函数
'''
功能：测试数据集的嫡值
输入：带标签值的数据集
输出：熵值
'''
Ent=CalcShannonEnt(myDat)
print(Ent,'显示测试数据集的嫡值')

#增加分类的种类后再测试嫡函数
'''
功能：测试改变后的数据集的嫡值
输入：带标签值的改变后的数据集
输出：熵值
'''
# myDat[0][-1]='maybe'
# print(myDat,'展示增加种类后的数据集')
# Ent=CalcShannonEnt(myDat)
# print(Ent,'显示改变后的测试数据集的嫡值')

#选择最佳的划分特征
'''
功能：选择最佳的数据集划分的特征
输入：带标签值的数据集
输出：最佳划分的特征
'''
feat=chooseBestFeatureToSplit(myDat)
print(feat,'选择最佳的数据集划分方式')

#创建决策树
'''
功能：创建一棵决策树采用字典的方式存储，其中最外面的字典的keys是最佳划分的特征名，
values是一个字典，其keys值是最佳划分的特征的属性名，其values是新的递归的一棵树。
输入：带标签值的数据集
输出：熵值
'''
mytree=createTree(myDat,labels)
print(mytree,'显示生成的决策树')
secondDict=mytree[mytree.keys()]
