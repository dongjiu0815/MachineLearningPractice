#__author__: dongj
#date: 2018/7/1
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import logRegres
import figure_plot

## step 1: load data
dataArr,labelMat=logRegres.loadDataSet()
test_x = dataArr
test_y = labelMat

## step 2: training...
print("step 2: training...")
print(logRegres.gradAscent(dataArr,labelMat))
optimalWeights=logRegres.gradAscent(dataArr,labelMat)

## step 3: testing
print("step 3: testing...")
accuracy = logRegres.testLogRegres(optimalWeights, test_x, test_y)

## step 4: show the result
print("step 4: show the result...")
print('The classify accuracy is: %.3f%%' % (accuracy * 100))
figure_plot.plotBestFit(optimalWeights.getA())