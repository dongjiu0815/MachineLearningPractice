#__author__: dongj
#date: 2018/7/31
def  LoadDataSet(filename):
    import numpy as np
    DataArr=[];LabelArr=[]
    fr=open('ex0.txt')
    lineArr=[]
    for line in fr.readlines():
        LineCur=line.strip().split('\t')
        lineArr.append(LineCur)
    lineArr=np.array(lineArr,dtype='float')
    DataArr=lineArr[:,:-1]
    LabelArr=lineArr[:,-1]
    return DataArr,LabelArr
def standRegRes_Lwlr(Testpoint,DataArr,LabelArr,k):
    import numpy as np
    DataArr=np.mat(DataArr)
    m,n=np.shape(DataArr)
    Q_array=np.ones(m)
    for i in range(m):
        error=Testpoint-DataArr[i,:]
        #print(error)
        Q_array[i]=np.exp(error*error.T/(-2.0*k**2))
    #print(Q_array)
    Q=np.diag(Q_array)
    #print(Q.shape)
    XQX=DataArr.T*(Q*DataArr)
    if np.linalg.det(XQX)==0:
        print('warning !')
        return
    #print(np.linalg.inv(XQX).shape)
    w=XQX.I*(DataArr.T*(Q*LabelArr))
    #print(np.dot(Testpoint,w).shape)
    return np.dot(Testpoint,w)
def Lwlr_test(xArr,yArr,k=0.03):
    import numpy as np
    xArr=np.mat(xArr,dtype='float')
    m,n=np.shape(xArr)
    y_pre=np.zeros(m)
    #print(y_pre.shape,'ffff')
    for i in range(m):
        y_pre[i]=standRegRes_Lwlr(xArr[i,:],xArr,np.mat(yArr).T,k)[0,0]
    #print(y_pre)
    # print(yArr)
    # print(y_pre-yArr)
    #计算决策系数
    u = sum((y_pre - np.mean(yArr, axis=0)) ** 2)
    v = sum((yArr - np.mean(yArr, axis=0)) ** 2)
    R=1-u/v
    return y_pre,R
