#__author__: dongj
#date: 2018/8/4
from 前向逐步回归_求导公式法 import LoadDataSet,StageWise,standRegRes
from figure_plot import plotBestFit
import numpy as np
## step 1: load data
print("step 1: load data...")
DataArr,LabelArr=LoadDataSet('abalone.txt')


WeightSet=StageWise(DataArr,LabelArr)
print(WeightSet)
WeightStage=WeightSet[-1:]
WeightStandRegRes=standRegRes(DataArr,LabelArr)
print('前向回归的权重与标准回归的权重的差',np.std(WeightStage-WeightStandRegRes),np.mean(WeightStage-WeightStandRegRes,axis=1))
plotBestFit(WeightSet)