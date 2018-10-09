# -*- coding: utf-8 -*-
# @Time    : 2018/9/27 21:14
# @Author  : 老湖
# @FileName: 网上实现版本.py
# @Software: PyCharm
# @qq    ：326883847
# coding:UTF-8
'''
Created on 2015年6月15日
@author: zhaozhiyong
'''

from numpy import *


def loadSimpleData():
    datMat = mat([[1., 2.1],
                  [2., 1.1],
                  [1.3, 1.],
                  [1., 1.],
                  [2., 1.]])
    classLabels = mat([1.0, 1.0, -1.0, -1.0, 1.0])
    return datMat, classLabels


def singleStumpClassipy(dataMat, dim, threshold, thresholdIneq):
    classMat = ones((shape(dataMat)[0], 1))
    # 根据thresholdIneq划分出不同的类，在'-1'和'1'之间切换
    if thresholdIneq == 'left':  # 在threshold左侧的为'-1'
        classMat[dataMat[:, dim] <= threshold] = -1.0
    else:
        classMat[dataMat[:, dim] > threshold] = -1.0

    return classMat


def singleStump(dataArr, classLabels, D):
    dataMat = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMat)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = zeros((m, 1))
    minError = inf
    for i in range(n):  # 对每一个特征
        # 取第i列特征的最小值和最大值，以确定步长
        rangeMin = dataMat[:, i].min()
        rangeMax = dataMat[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            # 不确定是哪个属于类'-1'，哪个属于类'1'，分两种情况
            for inequal in ['left', 'right']:
                threshold = rangeMin + j * stepSize  # 得到每个划分的阈值
                predictionClass = singleStumpClassipy(dataMat, i, threshold, inequal)
                errorMat = ones((m, 1))
                errorMat[predictionClass == labelMat] = 0
                weightedError = D.T * errorMat  # D是每个样本的权重
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictionClass.copy()
                    bestStump['dim'] = i
                    bestStump['threshold'] = threshold
                    bestStump['inequal'] = inequal

    return bestStump, minError, bestClasEst


def adaBoostTrain(dataArr, classLabels, G):
    weakClassArr = []
    m = shape(dataArr)[0]  # 样本个数
    # 初始化D，即每个样本的权重
    D = mat(ones((m, 1)) / m)
    aggClasEst = mat(zeros((m, 1)))

    for i in range(G):  # G表示的是迭代次数
        bestStump, minError, bestClasEst = singleStump(dataArr, classLabels, D)
        print('D:', D.T)
        # 计算分类器的权重
        alpha = float(0.5 * log((1.0 - minError) / max(minError, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print('bestClasEst:', bestClasEst.T)

        # 重新计算每个样本的权重D
        expon = multiply(-1 * alpha * mat(classLabels).T, bestClasEst)
        D = multiply(D, exp(expon))
        D = D / D.sum()

        aggClasEst += alpha * bestClasEst
        print('aggClasEst:', aggClasEst)
        aggErrors = multiply(sign(aggClasEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print('total error:', errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr


def adaBoostClassify(testData, weakClassify):
    dataMat = mat(testData)
    m = shape(dataMat)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(weakClassify)):  # weakClassify是一个列表
        classEst = singleStumpClassipy(dataMat, weakClassify[i]['dim'], weakClassify[i]['threshold'],
                                       weakClassify[i]['inequal'])
        aggClassEst += weakClassify[i]['alpha'] * classEst
        print(aggClassEst)
    return sign(aggClassEst)


if __name__ == '__main__':
    datMat, classLabels = loadSimpleData()
    weakClassArr = adaBoostTrain(datMat, classLabels, 30)
    print("weakClassArr:", weakClassArr)
    # test
    result = adaBoostClassify([1, 1], weakClassArr)
    print(result)
