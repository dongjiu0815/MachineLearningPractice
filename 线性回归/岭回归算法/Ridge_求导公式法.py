#__author__: dongj
#date: 2018/7/16
def LoadDataSet(filename):
    import  numpy as np
    DataArr=[];LabelArr=[]
    with open(filename) as fr:
        lineArr=[]
        for line in fr.readlines():
            lineCur=line.strip().split('\t')
            lineArr.append(lineCur)
    lineArr=np.array(lineArr)
    DataArr=lineArr[:,:-1]
    LabelArr=lineArr[:,-1]
    return DataArr,LabelArr
def RidgeRegres(DataArr,LabelArr,lam=0.2):
    import numpy as np
    DataArr=np.mat(DataArr)
    LabelArr=np.mat(LabelArr)
    m,n=np.shape(DataArr)
    xTx=DataArr.T*DataArr
    if np.linalg.det(xTx)==0.0:
        print('warning !')
        return
    # print(xTx.shape)
    # print(np.eye(n).shape)
    #print(LabelArr.shape)
    weight=((xTx+lam*np.eye(n)).I*(DataArr.T*LabelArr))
    #print(weight)
    return weight
def RidgeTest(xArr,yArr):
    import numpy as np
    xMat=np.mat(xArr,dtype='float')
    yMat=np.mat(yArr,dtype='float').T
    n=np.shape(xMat)[1]
    #进行数据归一化和标准话，否则不能够画出岭迹。
    ymeans=np.mean(yMat,0)
    xmeans=np.mean(xMat,0)
    xvars=np.var(xMat,0)
    yMat=yMat-ymeans
    xMat=(xMat-xmeans)/xvars
    numTestPts=30
    wMat=np.zeros((numTestPts,n),dtype='float')
    for i in range(numTestPts):
        ws=RidgeRegres(xMat,yMat,np.exp(i-10))
        #print(ws.T)
        wMat[i,:]=ws.T
    return wMat

