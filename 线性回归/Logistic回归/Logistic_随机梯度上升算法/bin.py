#__author__: dongj
#date: 2018/7/2
from logRegres_随机梯度上升法 import loadDataSet,stocGradAscent
from figure_plot import bestplotFit
datamat,labelmat=loadDataSet()
weights=stocGradAscent(datamat,labelmat)
bestplotFit(weights)
