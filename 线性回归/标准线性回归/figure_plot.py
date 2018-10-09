#__author__: dongj
#date: 2018/7/17
def plotBestFit(filename,weight):
    import matplotlib.pyplot as plt
    #from 线性回归_求导公式法 import loadDataSet
    from 批量梯度下降法求解线性回归 import loadDataSet
    import numpy as np
    dataArr,labelArr=loadDataSet(filename)
    dataArr=np.array(dataArr)
    labelArr=np.array(labelArr).reshape(200,1)
    m,n=np.shape(dataArr)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    # print('aaa')
    # print(dataArr[:,1].reshape(200,1).shape)
    # print(labelArr.shape)
    ax.scatter(dataArr[:,1].reshape(200,1),labelArr,c='b',s=10)
    #print('nnn')
    a=np.arange(dataArr[:,1].min(),dataArr[:,1].max(),0.1).reshape(1,10)
    #print('ccc')
    y_pre=np.array(weight[1]*a)+np.array(weight[0]*len(a))
    print('ddd')
    print(a)
    print(y_pre,'aaa')
    ax.plot(a.reshape(10,1),y_pre.reshape(10,1),c='r',linewidth=2, markersize=12)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('线性回归')
    plt.show()
