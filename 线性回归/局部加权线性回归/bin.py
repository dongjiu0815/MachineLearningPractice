#__author__: dongj
#date: 2018/7/18
#from Lwlr_求导公式法 import LoadDataSet,Lwlr_test
from figure_plot import plotBestFit
from Lwlr_梯度下降法 import LoadDataSet,Lwlr_test
DataArr,LabelArr=LoadDataSet('ex0.txt')
y_pre,R=Lwlr_test(DataArr,LabelArr,k=0.003)
#print('预测值为:',y_pre)
print('预测精度为:',R)
plotBestFit(DataArr,LabelArr,k=0.003)