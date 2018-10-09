#__author__: dongj
#date: 2018/7/20
def loadDataSet(filename):
    import numpy as np
    dataArr,labelArr,cur=[],[],[]
    with open(filename) as textfile:
        for line in textfile.readlines():
            cur.append(line.strip().split('\t'))
    cur=np.array(cur,dtype='float32')
    cur=np.random.shuffle(cur)
    dataArr,labelArr=cur[:,:-1],cur[:,-1]
    return dataArr,labelArr
def standRegRes_SGD(dataArr,labelArr,alpha=0.001,eps=0.01):
    import numpy as np
    dataArr=np.mat(dataArr)
    m,n=np.shape(dataArr)
    #weight=np.ones((n,1))
    weight=[[3],[2]]#结果非常受初值的影响
    # print(weight.shape)
    # print(dataArr.shape)
    count=0.0
    for i in range(m):
        count+=1
        error=np.dot(dataArr[i,:],weight)-labelArr[i]
        # print(weight.shape,'aaa')
        # print(dataArr[i].T.shape,'nnn')
        weight-=alpha*error[0,0]*dataArr[i,:].T
        if np.linalg.norm(error,ord=np.inf)<=eps:
            break
    weight=np.mat(weight)
    return weight,count