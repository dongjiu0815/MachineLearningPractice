#__author__: dongj
#date: 2018/7/18
def plotBestFit(DataArr,LabelArr,k=0.03):
    from Lwlr_梯度下降法 import standRegRes_Lwlr_BGD
    import numpy as np
    import matplotlib.pyplot as plt
    DataArr=np.mat(DataArr)
    m,n=np.shape(DataArr)
    y_pre=np.zeros(m)
    for i in range(m):
        y_pre[i]=standRegRes_Lwlr_BGD(DataArr[i,:],DataArr,np.mat(LabelArr).T,k)[0,0]
    #print(y_pre,'kkk')
    #print(DataArr[:,1])
    index=DataArr[:,1].argsort(0)
    #print(index)
    #xsrot=DataArr[index][:,0,:]可以用给如下的写法代替
    xsort=DataArr[index.flatten().A[0]]
    #print(xsrot)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(DataArr[:,1].flatten().A[0],np.mat(LabelArr).T.flatten().A[0],s=2,c='r')
    ax.plot(xsort[:,1],y_pre[index],'-',c='b')
    plt.show()