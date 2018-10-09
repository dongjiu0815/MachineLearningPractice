# -*- coding: utf-8 -*-
# @Time    : 2018/9/12 16:54
# @Author  : 老湖
# @FileName: PlattSMO.py
# @Software: PyCharm
# @qq    ：326883847
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd()))))
from 支持向量机.SMO算法.svmMLiA import selectJrand,clipAlpha
import numpy as np
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def kernelTrans(X, A, kTup):
    """
    Function：   核转换函数

    Input：      X：数据集
                A：某一行数据
                kTup：核函数信息

    Output： K：计算出的核向量
    """
    #获取数据集行列数
    m, n = np.shape(X)
    #初始化列向量
    K = np.mat(np.zeros((m, 1)))
    #根据键值选择相应核函数
    #lin表示的是线性核函数
    if kTup[0] == 'lin': K = X * A.T
    #rbf表示径向基核函数
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow * deltaRow.T
        #对矩阵元素展开计算，而不像在MATLAB中一样计算矩阵的逆
        K =  np.exp(K/(-1*kTup[1]**2))
    #如果无法识别，就报错
    else: raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    #返回计算出的核向量
    return K

class optStructkK:
    """
    Function：   存放运算中重要的值

    Input：      dataMatIn：数据集
                classLabels：类别标签
                C：常数C
                toler：容错率
                kTup：速度参数

    Output： X：数据集
                labelMat：类别标签
                C：常数C
                tol：容错率
                m：数据集行数
                b：常数项
                alphas：alphas矩阵
                eCache：误差缓存
                K：核函数矩阵
    """
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))

        """ 主要区分 """
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)
        """ 主要区分 """

def calcEkK(oS, k):
    """
    Function：   计算误差值E

    Input：      oS：数据结构
                k：下标

    Output： Ek：计算的E值
    """

    """ 主要区分 """
    #计算fXk，整个对应输出公式f(x)=w`x + b
    #fXk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k,:].T)) + oS.b
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T*oS.K[:, k] + oS.b)
    """ 主要区分 """

    #计算E值
    Ek = fXk - float(oS.labelMat[k])
    #返回计算的误差值E
    return Ek

def selectJK(i, oS, Ei):
    """
    Function：   选择第二个alpha的值

    Input：      i：第一个alpha的下标
                oS：数据结构
                Ei：计算出的第一个alpha的误差值

    Output： j：第二个alpha的下标
                Ej：计算出的第二个alpha的误差值
    """
    #初始化参数值
    maxK = -1; maxDeltaE = 0; Ej = 0
    #构建误差缓存
    oS.eCache[i] = [1, Ei]
    #构建一个非零列表，返回值是第一个非零E所对应的alpha值，而不是E本身
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    #如果列表长度大于1，说明不是第一次循环
    if (len(validEcacheList)) > 1:
        #遍历列表中所有元素
        for k in validEcacheList:
            #如果是第一个alpha的下标，就跳出本次循环
            if k == i: continue
            #计算k下标对应的误差值
            Ek = calcEkK(oS, k)
            #取两个alpha误差值的差值的绝对值
            deltaE = abs(Ei - Ek)
            #最大值更新
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        #返回最大差值的下标maxK和误差值Ej
        return maxK, Ej
    #如果是第一次循环，则随机选择alpha，然后计算误差
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEkK(oS, j)
    #返回下标j和其对应的误差Ej
    return j, Ej

def updateEkK(oS, k):
    """
    Function：   更新误差缓存

    Input：      oS：数据结构
                j：alpha的下标

    Output： 无
    """
    #计算下表为k的参数的误差
    Ek = calcEkK(oS, k)
    #将误差放入缓存
    oS.eCache[k] = [1, Ek]

def innerLK(i, oS):
    """
    Function：   完整SMO算法中的优化例程

    Input：      oS：数据结构
                i：alpha的下标

    Output： 无
    """
    #计算误差
    #如果标签与误差相乘之后在容错范围之外，且超过各自对应的常数值，则进行优化
    # 计算第一样本的误差
    Ei = calcEkK(oS, i)
    # print('第', i, '个样本的误差为', Ei)
    # print('第', i, '个样本的alphas为', oS.alphas[i])
    # print('第', i, '个样本的标签为', oS.labelMat[i])
    # 如果第一个参数使得误差足够大，并且相应的alphas属于0到c之间那么就作为第一个alphas
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        #启发式选择第二个alpha值
        j, Ej = selectJK(i, oS, Ei)
        #利用copy存储刚才的计算值，便于后期比较
        alphaIold = oS.alphas[i].copy(); alpahJold = oS.alphas[j].copy();
        #保证alpha在0和C之间
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS. alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        #如果界限值相同，则不做处理直接跳出本次循环
        if L == H: print("L==H"); return 0

        """ 主要区分 """
        #最优修改量，求两个向量的内积（核函数）
        #eta = 2.0 * oS.X[i, :]*oS.X[j, :].T - oS.X[i, :]*oS.X[i, :].T - oS.X[j, :]*oS.X[j, :].T
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        """ 主要区分 """

        #如果最优修改量大于0，则不做处理直接跳出本次循环，这里对真实SMO做了简化处理
        if eta >= 0: print("eta>=0"); return 0
        #计算新的alphas[j]的值
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        #对新的alphas[j]进行阈值处理
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        #更新误差缓存
        updateEkK(oS, j)
        #如果新旧值差很小，则不做处理跳出本次循环
        if (abs(oS.alphas[j] - alpahJold) < 0.00001): print("j not moving enough"); return 0
        #对i进行修改，修改量相同，但是方向相反
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alpahJold - oS.alphas[j])
        #更新误差缓存
        updateEkK(oS, i)

        """ 主要区分 """
        #更新常数项
        #b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :]*oS.X[i, :].T - oS.labelMat[j] * (oS.alphas[j] - alpahJold) * oS.X[i, :]*oS.X[j, :].T
        #b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :]*oS.X[j, :].T - oS.labelMat[j] * (oS.alphas[j] - alpahJold) * oS.X[j, :]*oS.X[j, :].T
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alpahJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (oS.alphas[j] - alpahJold) * oS.K[j, j]
        """ 主要区分 """

        #谁在0到C之间，就听谁的，否则就取平均值
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[i]): oS.b = b2
        else: oS.b = (b1 + b2) / 2.0
        #成功返回1
        return 1
    #失败返回0
    else: return 0

def smoPK(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin', 0)):
    """
    Function：   完整SMO算法

    Input：      dataMatIn：数据集
                classLabels：类别标签
                C：常数C
                toler：容错率
                maxIter：最大的循环次数
                kTup：速度参数

    Output： b：常数项
                alphas：数据向量
    """
    #新建数据结构对象
    oS = optStructkK(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
    #初始化迭代次数
    iter = 0
    #初始化标志位
    entireSet = True; alphaPairsChanged = 0
    #终止条件：迭代次数超限、遍历整个集合都未对alpha进行修改
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        #根据标志位选择不同的遍历方式
        if entireSet:
            #遍历任意可能的alpha值
            for i in range(oS.m):
                #选择第二个alpha值，并在可能时对其进行优化处理
                alphaPairsChanged += innerLK(i, oS)
                print("fullSet, iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged))
            #迭代次数累加
            iter += 1
        else:
            #得出所有的非边界alpha值
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            #遍历所有的非边界alpha值
            for i in nonBoundIs:
                #选择第二个alpha值，并在可能时对其进行优化处理
                alphaPairsChanged += innerLK(i, oS)
                print("non-bound, iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged))
            #迭代次数累加
            iter += 1
        #在非边界循环和完整遍历之间进行切换
        if entireSet: entireSet = False
        elif (alphaPairsChanged == 0): entireSet =True
        print("iteration number: %d" % iter)
    #返回常数项和数据向量
    return oS.b, oS.alphas

def testRbf(k1 = 1.3):
    """
    Function：   利用核函数进行分类的径向基测试函数

    Input：      k1：径向基函数的速度参数

    Output： 输出打印信息
    """
    #导入数据集
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    #调用Platt SMO算法
    b, alphas = smoPK(dataArr, labelArr, 200, 0.00001, 10000, ('rbf', k1))
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
        kernelEval = kernelTrans(sVs, datMat[i,:], ('rbf', k1))
        #仅用支持向量预测分类,这是由于非支持向量的alphas为0，所以相乘后的结果为零,所以不用考虑那些样本。
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        #预测分类结果与标签不符则错误计数加一
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    #打印输出分类错误率
    print("the training error rate is: %f" % (float(errorCount)/m))
    #导入测试数据集
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    #初始化误差计数
    errorCount = 0
    #初始化数据矩阵和标签向量
    datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    #获取数据集行列值
    m, n = np.shape(datMat)
    #遍历每一行，利用核函数对测试集进行分类
    for i in range(m):
        #利用核函数转换数据
        kernelEval = kernelTrans(sVs, datMat[i,:], ('rbf', k1))
        #仅用支持向量预测分类
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        #预测分类结果与标签不符则错误计数加一
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    #打印输出分类错误率
    print("the test error rate is: %f" % (float(errorCount)/m))

'''#######********************************
Non-Kernel VErsions below
'''  #######********************************
class optStructK:
    def __init__(self, dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))  # first column is valid flag

def calcEk(oS, k):#计算误差EK
    # print(oS.alphas.shape,'alpah的维度')
    #print(oS.labelMat, 'labelMat的维度')
    # print(np.multiply(oS.alphas, oS.labelMat).T.shape,'显示维度')
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.X*oS.X[k, :].reshape(-1,1)) + oS.b)
    #print(fXk,'显示fXk')
    Ek = fXk - float(oS.labelMat[k])
    #print(Ek,'显示Ek')
    return Ek

def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w

def updateEk(oS,k):#更新Ek after any alpha has changed update the new value in the cache
    Ek=calcEk(oS,k)
    oS.eCache[k]=[1,Ek]

#选择第二个变量j和其误差
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
                maxK = k
                maxDeltaE = deltaE
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

def innerL(i, oS):
    Ei = calcEk(oS, i)
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
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)  # added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])  # update i by the same amount as j
        updateEk(oS, i)  # added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter):  # full Platt SMO
    oS = optStructK(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True;
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:  # go over all
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
