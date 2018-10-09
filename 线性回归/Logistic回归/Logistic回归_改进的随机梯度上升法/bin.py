#__author__: dongj
#date: 2018/7/2
from logRegres_改进随机梯度上升法 import loadDataSet,stocGradAscentImprove
from figure_plot import bestplotFit
datamat,labelmat=loadDataSet()
weights=stocGradAscentImprove(datamat,labelmat)
bestplotFit(weights)
