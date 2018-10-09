# -*- coding:UTF-8 -*-
#__author__: dongj
#date: 2018/7/16
#from Ridge_求导公式法 import LoadDataSet,RidgeTest
#from Ridge_梯度下降法 import LoadDataSet,RidgeTest
#from figure_plot import plotBestFit
from Ridge_爬虫交叉验证版 import crossValidation,setDataCollect

# DataArr,LabelArr=LoadDataSet('abalone.txt')
# weight=RidgeTest(DataArr,LabelArr)
# #print(weight)
# plotBestFit(weight)
lgx=[];lgy=[]
setDataCollect(lgx,lgy)
crossValidation(lgx,lgy)