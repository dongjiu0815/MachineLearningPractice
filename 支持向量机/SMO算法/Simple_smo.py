# -*- coding: utf-8 -*-
# @Time    : 2018/9/11 14:29
# @Author  : 老湖
# @FileName: Simple_svm.py
# @Software: PyCharm
# @qq    ：326883847
from svmMLiA import selectJrand,clipAlpha

#dataMatIN为输入数据x矩阵，classlabels为标签，C为正则化系数，toler为容许误差，maxIter为最大迭代次数
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    import numpy as np
    dataMatrix=np.mat(dataMatIn);labelMat=np.mat(classLabels).transpose()
    #print(labelMat.shape,'显示维度')
    b=0;m,n=np.shape(dataMatrix)
    alphas=np.mat(np.zeros((m,1)))#初始化alpha的值
    iter=0
    while(iter<maxIter):
        alphaPairsChanged=0#更新成功标志
        for i in range(m):
            fXi=float(np.multiply(alphas,labelMat).T*\
                      (dataMatrix*dataMatrix[i,:].T))+b
            Ei=fXi-float(labelMat[i])#计算第一个变量的误差
            if ((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or \
                    ((labelMat[i]*Ei>toler) and \
                    (alphas[i]>0)):
                j=selectJrand(i,m)#随机选择第二个变量
                fXj=float(np.multiply(alphas,labelMat).T*\
                    (dataMatrix*dataMatrix[j,:].T))+b#计算第二个变量的误差
                Ej=fXj-float(labelMat[j])
                alphaIold=alphas[i].copy()#保存旧的alpha值
                alphaJold=alphas[j].copy()#保存旧的alpha值
                if(labelMat[i]!=labelMat[j]):
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])
                if L==H:print('L==H');continue
                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-\
                    dataMatrix[i,:]*dataMatrix[i,:].T-\
                    dataMatrix[j,:]*dataMatrix[j,:].T#计算2k12-k11-k22
                if eta>=0:print('eta>=0');continue
                alphas[j]-=labelMat[j]*(Ei-Ej)/eta#计算alpha1^new,unc
                alphas[j]=clipAlpha(alphas[j],H,L)#计算alpha1^new
                if (abs(alphas[j]-alphaJold)<0.00001):print\
                    ('j not moving enough');continue#设置终止条件
                alphas[i]+=labelMat[j]*labelMat[i]*\
                           (alphaJold-alphas[j])#计算alpha2^new
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaJold)*\
                    dataMatrix[i,:]*dataMatrix[i,:].T-\
                    labelMat[j]*(alphas[j]-alphaJold)*\
                    dataMatrix[i,:]*dataMatrix[j,:].T#计算b1^new
                b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*\
                    dataMatrix[i,:]*dataMatrix[j,:].T-\
                    labelMat[j]*(alphas[j]-alphaJold)*\
                    dataMatrix[j,:]*dataMatrix[j,:].T#计算b2^new
                if (0<alphas[i])and(C>alphas[i]):b=b1
                elif(0<alphas[i]) and (C>alphas[i]):b=b2
                else:b=(b1+b2)/2.0#确定最终的b
                alphaPairsChanged+=1
                print('iter: %d i:%d, pairs changed %d'%\
                      (iter,i,alphaPairsChanged))
                if (alphaPairsChanged==0):iter+=1
                else:iter=0
                print('iteration number :%d'%iter)
            return b,alphas
