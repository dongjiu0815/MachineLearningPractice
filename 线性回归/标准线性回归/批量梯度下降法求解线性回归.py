#__author__: dongj
#date: 2018/7/18
def loadDataSet(filename):
    import numpy as np
    dataArr,labelArr,cur=[],[],[]
    with open(filename) as textfile:
        for line in textfile.readlines():
            cur.append(line.strip().split('\t'))
    cur=np.array(cur,dtype='float32')
    dataArr,labelArr=cur[:,:-1],cur[:,-1]
    return dataArr,labelArr
def standRegRes_BGD(dataArr,labelArr,itermax=500,alpha=0.001,eps=0.15):
    import numpy as np
    dadtaArr=np.mat(dataArr)
    m,n=np.shape(dataArr)
    weight=np.ones((n,1))
    #print(weight.shape)
    labelArr=labelArr[:,np.newaxis]
    sumerr=0.0
    count=0
    for k in range(itermax):
        count+=1
        error = np.dot(dataArr, weight)-labelArr
        # #print(error)
        # for j in range(n):
        #     for i in range(m):
        #         sumerr+=float(error[i])*dataArr[i,j]
        #         # print(sumerr)
        #     weight[j]=weight[j]-alpha*sumerr
        #     sumerr=0.0
        weight=weight-alpha*np.dot(dataArr.T,error)
        if np.linalg.norm(error,ord=np.inf)<=eps:
            break
    weight=np.mat(weight)
    return weight,count