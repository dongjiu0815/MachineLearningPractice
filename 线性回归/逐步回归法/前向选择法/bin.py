#__author__: dongj
#date: 2018/8/10
from 前向选择法 import loadData,ForwardStepwise,ModelSelect,StandardRegres,rssError
#from figure_plot import
import numpy as np

# step 1: load data
print('step 1: load data...')
DataArr,LabelArr=loadData(r'abalone.txt')
#print(LabelArr.shape)
#print(LabelArr,LabelArr.shape)

# step 2: 采用前向选择法生成n个模型
print("step 2: Generating model...")
#opts参数选择optimizeType：DerivationFormula，BatchGradDescent，StocGradDescent，smoothStocGradDescent，MiniBatchGradDescent

opts = {'alpha': 0.01, 'IterMax': 500, 'optimizeType': 'smoothStocGradDescent','b':10,'eps':0.01}
Model= ForwardStepwise(DataArr,LabelArr, opts)
#print(Model,'模型')
# Model4=Model[3]
# xData=Model4[1]
# yData=Model4[2]
# y_pre4=Model4[3]
# score4=Model4[4]
# print(xData.shape,'x数据')
# print(yData.shape,'y数据')
# print(score4,'预测的分值')
#step 3: model select
print("step 3: model select...")
BestModel= ModelSelect(Model,Method='CP_Score')

# step 4: show the result
print("step 4: show the result...")
accuracy=BestModel[4]
print(accuracy,'准确率')
print('The classify accuracy is: %.3f%%' % (accuracy * 100))

#step 5:划分测试集和训练集
DataMat=BestModel[1]
m,n=np.shape(DataMat)
LabelMat=BestModel[2]
numindex=range(m)
np.random.shuffle(list(numindex))
xTrain,yTrain=[],[]
xTest,yTest=[],[]
for i in range(m):
    if i<m*0.8:
        xTrain.append(DataMat[numindex[i]])
        yTrain.append(LabelMat[numindex[i]])
    else:
        xTest.append(DataMat[numindex[i]])
        yTest.append(LabelMat[numindex[i]])
yTrain=np.array(yTrain)
yTest=np.array(yTest)
weightTrain=StandardRegres(xTrain,yTrain,opts)
y_preTest=np.dot(xTest,weightTrain)
print(y_preTest.shape,'预测值的维度')
print(yTest.shape,'测试值的维度')
print('测试准确率为',rssError(y_preTest.flatten(),yTest))
#showLogRegres(optimalWeights, train_x, train_y)