#__author__: dongj
#date: 2018/8/3
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
def RidgeRegres_BGD(DataArr,LabelArr,alpha=0.001,itermax=500,eps=0.001,lam=0.8):
    import numpy as np
    DataArr=np.mat(DataArr)
    m,n=np.shape(DataArr)
    xTx=DataArr.T*DataArr
    weight=np.ones((n,1))
    for i in range(itermax):
        error=DataArr*weight-LabelArr
        print(error)
        weight=weight-2*alpha*(DataArr.T*error+lam*weight)
        if(np.linalg.norm(error,ord=np.inf)<=eps):
            break
    weight=np.mat(weight)
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
        ws=RidgeRegres_BGD(xMat,yMat,lam=np.exp(i-10))
        #print(ws.T)
        wMat[i,:]=ws.T
    return wMat
