#__author__: dongj
#date: 2018/7/17
import sys,os
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
from figure_plot import plotBestFit
from 线性回归_求导公式法 import loadDataSet
from 线性回归_求导公式法 import standRegRes
from 批量梯度下降法求解线性回归 import standRegRes_BGD
from 随机梯度下降法求解线性回归 import standRegRes_SGD
from 小批量梯度下降法求解线性回归 import standRegRes_MBGD
from 改进的随机梯度下降算法求解线性回归 import standRegRes_ImproveSGD


#dataArr,labelArr=loadDataSet('housePricesPredicting.txt')
xArr,yArr=loadDataSet('ex0.txt')
# print(xArr,'aaa')
# print(yArr,'bbb')
# weight=standRegRes(xArr,yArr)#采用标准的公式法求解标准线性规划
#(weight,count)=standRegRes_BGD(xArr,yArr)#采用梯度下降法来求解标准线性规划
#(weight,count)=standRegRes_SGD(xArr,yArr)#采用随机梯度下降法求解标准线性规划
#(weight,count)=standRegRes_MBGD(xArr,yArr)
(weight,count)=standRegRes_ImproveSGD(xArr,yArr)
print('回归系数是：',weight)
print('迭代次数是：',count)
print(weight,'e2')
filename='ex0.txt'#
plotBestFit(filename,weight)
y_pre=np.mat(xArr)*np.mat(weight).T
print(np.corrcoef(y_pre.T,yArr))#corrcoef用于计算相关系数查看拟合的情况