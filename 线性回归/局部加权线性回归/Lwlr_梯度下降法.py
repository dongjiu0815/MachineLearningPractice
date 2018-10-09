#__author__: dongj
#date: 2018/8/3
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
def standRegRes_Lwlr_BGD(Testpoint,DataArr,LabelArr,alpha=0.001,eps=0.15,k=0.03,itermax=200):
    import numpy as np
    DataArr=np.mat(DataArr)
    m,n=np.shape(DataArr)
    weight = np.ones((n, 1))
    #print(weight.shape,'lll')
    #LabelArr =np.mat(LabelArr)
    Q=np.eye(m)
    for i in range(m):
        differ=DataArr[i,:]-Testpoint
        #print(differ.shape)
        Q[i,i]=np.exp((differ*differ.T)/(-2.0*k**2))
    for j in range(itermax):
        #print('kuku')
        error = np.dot(DataArr, weight)-LabelArr
        #print(error.shape,'www')
        weight=weight-2*alpha*DataArr.T*Q*error
        if np.linalg.norm(error,ord=np.inf)<=eps:
            print('停止')
            break
    weight=np.mat(weight)
    #print(np.dot(Testpoint,weight).shape)
    return np.dot(Testpoint,weight)
def Lwlr_test(xArr,yArr,k=0.03):
    import numpy as np
    xArr=np.mat(xArr,dtype='float')
    m,n=np.shape(xArr)
    y_pre=np.zeros(m)
    #print(y_pre.shape,'ffff')
    for i in range(m):
        y_pre[i]=standRegRes_Lwlr_BGD(xArr[i,:],xArr,np.mat(yArr).T,k)[0,0]
    print(y_pre)
    # print(yArr)
    # print(y_pre-yArr)
    #计算决策系数
    u=sum((y_pre-np.mean(yArr,axis=0))**2)
    v=sum((yArr-np.mean(yArr,axis=0))**2)
    R=1-u/v
    return y_pre,R

