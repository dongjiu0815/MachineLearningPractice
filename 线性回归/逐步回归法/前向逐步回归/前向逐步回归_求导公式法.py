#__author__: dongj
#date: 2018/8/4
def  LoadDataSet(filename):
    import numpy as np
    DataArr=[];LabelArr=[]
    fr=open('abalone.txt')
    lineArr=[]
    for line in fr.readlines():
        LineCur=line.strip().split('\t')
        lineArr.append(LineCur)
    lineArr=np.array(lineArr,dtype='float')
    DataArr=lineArr[:,:-1]
    LabelArr=lineArr[:,-1]
    return DataArr,LabelArr

def standRegRes(dataArr,labelArr):
    import numpy as np
    dataArr=np.mat(dataArr)
    labelArr=np.mat(labelArr).T
    m=np.shape(dataArr)[0]
    if np.linalg.det(dataArr.T*dataArr)==0.0:
        print('this matrix is singular,cannot do inverse')
    else:
        # if (dataArr[:,0].all()==np.shape(dataArr)[0]*[1]).all():
        #     return np.dot(np.linalg.inv(dataArr.T*dataArr),dataArr.T*labelArr)
        # else:
        #     np.hstack((np.array(m*[1]).reshape(m,1),dataArr))
        #     return np.dot(np.linalg.inv(dataArr.T*dataArr),dataArr.T*labelArr)
        return np.dot(np.linalg.inv(dataArr.T * dataArr), dataArr.T * labelArr)

def AIC_Score(y_pre,yArr,m,n):
    import numpy as np
    SSR = sum((y_pre - yArr) ** 2)
    return 2*n+m*np.log(SSR/m)

def BIC_Score(y_pre,yArr,m,n):
    import numpy as np
    SSR=sum((y_pre-yArr)**2)
    sigma=np.std(y_pre-yArr)
    return (SSR+n*sigma*np.log(m))/m

#调整决定系数R^2
def R2_Score(y_pre,yArr,m,n):
    import numpy as np
    SSR=sum((y_pre-yArr)**2)
    SST=sum((yArr-np.mean(yArr,axis=0))**2)
    return 1-((SSR/(m-n-1))/(SST/(m-1)))

def CP_Score(y_pre,yArr,m,n):
    import numpy as np
    SSR = sum((y_pre - yArr) ** 2)
    sigma = np.std(y_pre - yArr)
    return (SSR+2*n*sigma)/m

def rssError(y_pre,yArr):
    return sum((y_pre-yArr)**2)


def StageWise(DataArr,LableArr,eps=0.005,itermax=1000):
    import numpy as np
    xArr=np.mat(DataArr)
    yArr=np.mat(LableArr).T
    yMeans=np.mean(yArr,axis=0)
    xMeans=np.mean(xArr,axis=0)
    xVars=np.var(xArr,axis=0)
    xMat=(xArr-xMeans)/xVars
    yMat=yArr-yMeans
    m,n=np.shape(xMat)
    WeightSet=np.zeros((itermax,n))
    Weight=np.zeros((n,1))
    WeightTest,WeightMax=Weight.copy(),Weight.copy()
    for i in range(itermax):
        #print(Weight)
        lowestError = np.inf
        for j in range(n):
            for sign in [-1,1]:
                WeightTest=Weight.copy()
                WeightTest[j,0]+=sign*eps
                yTest=xMat*WeightTest
                rssE=rssError(yTest.A,yMat.A)
                if rssE<lowestError:
                    lowestError=rssE
                    WeightMax=WeightTest
        Weight=WeightMax.copy()
        WeightSet[i,:]=Weight.T
    return WeightSet








