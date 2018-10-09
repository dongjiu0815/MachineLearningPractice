#__author__: dongj
#date: 2018/8/10
from 逻辑回归 import loadData,trainLogRegres,testLogRegres
from figure_plot import showLogRegres
## step 1: load data
print("step 1: load data...")
train_x, train_y = loadData()
test_x = train_x
test_y = train_y

## step 2: training...
print("step 2: training...")
opts = {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'smoothStocGradDescent'}
optimalWeights = trainLogRegres(train_x, train_y, opts)

## step 3: testing
print("step 3: testing...")
accuracy = testLogRegres(optimalWeights, test_x, test_y)

## step 4: show the result
print("step 4: show the result...")
print('The classify accuracy is: %.3f%%' % (accuracy * 100))
showLogRegres(optimalWeights, train_x, train_y)