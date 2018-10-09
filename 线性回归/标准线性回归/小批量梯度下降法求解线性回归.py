#__author__: dongj
#date: 2018/7/20
def loadDataSet(filename):
    import numpy as np
    dataArr,labelArr,cur=[],[],[]
    with open(filename) as textfile:
        for line in textfile.readlines():
            cur.append(line.strip().split('\t'))
    cur=np.array(cur,dtype='float32')
    dataArr,labelArr=cur[:,:-1],cur[:,-1]
    return dataArr,labelArr

def standRegRes_MBGD(dataArr,labelArr,alpha=0.001,eps=0.0001,b=10,weight=[[3],[1.6]]):
    import numpy as np
    dataArr=np.mat(dataArr)
    m,n=dataArr.shape
    #weight=np.ones((n,1))
    #weight=[[3],[3]]
    weight=np.array(weight,dtype=float).reshape(n,1)
    count=0.0
    print(labelArr.shape)
    for j in range(n):
        #count += 1
        error = np.dot(dataArr, weight) - labelArr[:,np.newaxis]
        print(error.shape)
        SumNum=0
        if isinstance(b,(int)):
            if b<m:
                pass
            else:
                print('超出了样本数,请重新输入')
        elif b.split('.')==0:
            b=b*m
        else:
            print('请重新输入b,b要么是0到1之间的小数,要么是小于样本数的整数')

        for i in range(j*b,b*j+b):
            count += 1
            SumNum+=dataArr[i,j]*error[i,0]
        weight-=alpha*SumNum
        if np.linalg.norm(error, ord=np.inf) < eps:
            break
    weight = np.mat(weight)
    return weight,count
