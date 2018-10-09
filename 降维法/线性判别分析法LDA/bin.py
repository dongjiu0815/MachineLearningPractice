# -*- coding: utf-8 -*-
# @Time    : 2018/8/28 19:12
# @Author  : 老湖
# @FileName: bin.py
# @Software: PyCharm
# @qq    ：326883847
from 线性判别分析LDA import LoadData,LDA_class
from figure_plot import plotBestFigure

#加载数据
DataMat,LabelMat=LoadData('watermelon_3a.txt')
#print(DataMat,'x')
#print(LabelMat,'Y')

#采用线性判别分析法求解
weight=LDA_class(DataMat,LabelMat,2)
weight=weight.flatten()
print(weight,'展示权重')
# print(DataMat,'展示数据DataMat')
#画出最后的图形
plotBestFigure(DataMat,LabelMat,weight)
