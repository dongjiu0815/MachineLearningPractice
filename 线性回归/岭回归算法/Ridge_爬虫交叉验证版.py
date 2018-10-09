#__author__: dongj
#date: 2018/8/10
from time import sleep
import json
import urllib.request


def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (
        myAPIstr, setNum)
    pg = urllib.request.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" %
                          (yr, numPce, newFlag, origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print('problem with item %d' % i)


def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


def rssError(y_pre,yArr):
    return sum((y_pre-yArr)**2)

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

def crossValidation(xArr,yArr,numVal=10):
    import numpy as np
    m=len(yArr)
    print(m)
    ListIndex=range(m)
    ErrorCv=np.zeros((numVal,30))
    for i in range(numVal):
        xTest=[];xTrain=[]
        yTest=[];yTrain=[]
        np.random.shuffle(list(ListIndex))
        for j in range(m):
            if j<m*0.9:
                xTrain.append(xArr[ListIndex[j]])
                yTrain.append(yArr[ListIndex[j]])
            else:
                xTest.append(xArr[ListIndex[j]])
                yTest.append(yArr[ListIndex[j]])
        WeightSet=RidgeTest(xTrain,yTrain)
        #WeightSet=np.mat(WeightSet,dtype='float')
        for k in range(30):
            xTest=np.mat(xTest,dtype='float');xTrain=np.mat(xTrain,dtype='float')
            xTrainMean=np.mean(xTrain,0)
            xTrainVar=np.var(xTrain,0)
            xTestMat=(xTest-xTrainMean)/xTrainVar#将测试集像训练集一样标准化
            xTestMat=np.mat(xTestMat,dtype='float')
            #print(WeightSet[j, :])
            #print(WeightSet[j, :].reshape((8,1)))
            #k=np.ones((len(yTrain),1))
            print(xTestMat.shape)
            print(xTrain.shape)
            yTest_pre=xTestMat*np.mat(WeightSet[k,:]).T+np.mean(yTrain,axis=0,dtype='float')
            ErrorCv[i,k]=rssError(yTest_pre.T.A,yTest.T)
    ErrorCvMean=np.mean(ErrorCv,0)
    ErrorCvMeanMin=np.min(ErrorCvMean)
    BestWeight=WeightSet[np.nonzero(ErrorCvMean==ErrorCvMeanMin)]
    xMat=np.mat(xArr);yMat=np.mat(yArr)
    xMatMean=np.mean(xMat,0)
    xMatVar=np.var(xMat,0)
    ExpWeight=BestWeight/xMatVar
    print('the best model from Ridge Regression is:\n',ExpWeight)
    print('with constant term:',-1*sum(np.multiply(xMatMean,ExpWeight))+np.mean(yMat))