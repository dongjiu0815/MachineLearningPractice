# -*- coding: utf-8 -*-
# @Time    : 2018/9/17 18:36
# @Author  : 老湖
# @FileName: SMO算法实现手写识别问题.py
# @Software: PyCharm
# @qq    ：326883847
import numpy as np
#import sys,os
# sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd()))))
# from 支持向量机.SMO算法.PlattSMO import smoP,kernelTrans
# print(sys.path)
def img2vector(filename):
    """
    Function：   32*32图像转换为1*1024向量

    Input：      filename：文件名称字符串

    Output： returnVect：转换之后的1*1024向量
    """
    #初始化要返回的1*1024向量
    returnVect = np.zeros((1, 1024))
    #打开文件
    fr = open(filename)
    #读取文件信息
    for i in range(32):
        #循环读取文件的前32行
        lineStr = fr.readline()
        for j in range(32):
            #将每行的头32个字符存储到要返回的向量中
            returnVect[0, 32*i+j] = int(lineStr[j])
    #返回要输出的1*1024向量
    return returnVect

def loadImages(dirName):
    """
    Function：   加载图片

    Input：      dirName：文件路径

    Output： trainingMat：训练数据集
                hwLabels：数据标签
    """
    from os import listdir
    #初始化数据标签
    hwLabels = []
    #读取文件列表
    trainingFileList = listdir(dirName)
    #读取文件个数
    m = len(trainingFileList)
    #初始化训练数据集
    trainingMat = np.zeros((m,1024))
    #填充数据集
    for i in range(m):
        #遍历所有文件
        fileNameStr = trainingFileList[i]
        #提取文件名称
        fileStr = fileNameStr.split('.')[0]
        #提取数字标识
        classNumStr = int(fileStr.split('_')[0])
        #数字9记为-1类
        if classNumStr == 9: hwLabels.append(-1)
        #其他数字记为+1类
        else: hwLabels.append(1)
        #提取图像向量，填充数据集
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    #返回数据集和数据标签
    return trainingMat, hwLabels

def selectJrand(i,m):
    import numpy as np
    j=i
    while(j==i):
        j=int(np.random.uniform(0,m))
    return j
def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

def calcEk(oS, k):
    # print(oS.alphas.shape,'alpah的维度')
    #print(oS.labelMat, 'labelMat的维度')
    # print(np.multiply(oS.alphas, oS.labelMat).T.shape,'显示维度')
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.X*oS.X[k, :].reshape(-1,1)) + oS.b)
    #print(fXk,'显示fXk')
    Ek = fXk - float(oS.labelMat[k])
    #print(Ek,'显示Ek')
    return Ek

def selectJ(i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
    maxK = -1;
    maxDeltaE = 0;
    Ej = 0
    oS.eCache[i] = [1, Ei]  # set valid #choose the alpha that gives the maximum delta E
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    #print('显示所有的误差不为零的索引',validEcacheList)
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:  # loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue  # don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            #print('显示判断中的deltaE的值',deltaE)
            if (deltaE > maxDeltaE):
                maxK = k;
                maxDeltaE = deltaE;
                Ej = Ek
        # print('选择的第二个变量为：',maxK)
        # print('选择的第二个变量的误差为：',Ej)
        # print('选择的第二个变量的alphas的值为：',oS.alphas[maxK])
        # print('选择的第二个变量的标签值为：',oS.labelMat[maxK])
        return maxK, Ej
    else:  # in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
        # print(j, '显示第一次任意随机选取的j的大小')
        # print(Ej, '显示第一次任意随机选取的j的误差')
        # print(oS.alphas[j],'显示第一次随机选取的j的alphas的值')
        # print(oS.labelMat[j], '显示第一次随机选取的j的标签值')
    return j, Ej

def updateEk(oS, k):  # after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def innerL(i, oS):#确定了第一个样本和alphas
    #计算第一样本的误差
    Ei = calcEk(oS, i)
    # print('第',i,'个样本的误差为',Ei)
    # print('第',i,'个样本的alphas为',oS.alphas[i])
    # print('第',i,'个样本的标签为',oS.labelMat[i])
    #如果第一个参数使得误差足够大，并且相应的alphas属于0到c之间那么就作为第一个alphas
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)  # this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy();
        alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H: print("L==H"); return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]  # changed for kernel
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)  # added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])  # update i by the same amount as j
        updateEk(oS, i)  # added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def kernelTrans(X, A, kTup):  # calc the kernel or transform data to a higher dimensional space
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T  # linear kernel
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1] ** 2))  # divide in NumPy is element-wise not matrix like Matlab
    else:
        raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))  # first column is valid flag
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)

def smoPK(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):  # full Platt SMO
    oS = optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(), C, toler,kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    #print(oS.labelMat,'显示标签值')
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:  # go over all
            #首先遍历所有的样本，选择任何可能的alphas
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:  # go over non-bound (railed) alphas
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas
def testDigits(kTup = ('rbf',10)):
    """
    Function：   手写数字分类函数

    Input：      kTup：核函数采用径向基函数

    Output： 输出打印信息
    """
    #导入数据集
    dataArr, labelArr = loadImages('trainingDigits')
    #调用Platt SMO算法
    b, alphas = smoPK(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    #初始化数据矩阵和标签向量
    datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    #记录支持向量序号
    svInd = np.nonzero(alphas.A > 0)[0]
    #读取支持向量
    sVs = datMat[svInd]
    #读取支持向量对应标签
    labelSV = labelMat[svInd]
    #输出打印信息
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    #获取数据集行列值
    m, n = np.shape(datMat)
    #初始化误差计数
    errorCount = 0
    #遍历每一行，利用核函数对训练集进行分类
    for i in range(m):
        #利用核函数转换数据
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        #仅用支持向量预测分类
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        #预测分类结果与标签不符则错误计数加一
        if np.sign(predict)!=np.sign(labelArr[i]): errorCount += 1
    #打印输出分类错误率
    print("the training error rate is: %f" % (float(errorCount)/m))
    #导入测试数据集
    dataArr,labelArr = loadImages('testDigits')
    #初始化误差计数
    errorCount = 0
    #初始化数据矩阵和标签向量
    datMat=np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    #获取数据集行列值
    m,n = np.shape(datMat)
    #遍历每一行，利用核函数对测试集进行分类
    for i in range(m):
        #利用核函数转换数据
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        #仅用支持向量预测分类
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        #预测分类结果与标签不符则错误计数加一
        if np.sign(predict)!=np.sign(labelArr[i]): errorCount += 1
    #打印输出分类错误率
    print("the test error rate is: %f" % (float(errorCount)/m))
