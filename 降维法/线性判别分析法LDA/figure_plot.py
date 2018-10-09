# -*- coding: utf-8 -*-
# @Time    : 2018/8/28 19:13
# @Author  : 老湖
# @FileName: figure_plot.py
# @Software: PyCharm
# @qq    ：326883847
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
def plotBestFigure(DataArr,LabelArr,w):
    m,n=np.shape(DataArr)
    x=np.linspace(DataArr[:,0].min(),DataArr[:,0].max(),100)
    z=-(w[0]/w[1])*x
    # print(x,'显示x数据')
    # print(z,'显示y数据')
    # print(n,'展示n的值')
    if n==2:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        LabelSet = list(set(LabelArr))
        DataSet = {}
        print('标签的类别有如下几种:', LabelSet)
        for i in LabelSet:
            DataSet[i] = DataArr[LabelArr == int(i)]
        # print(DataSet[0],'xttttt的数据集')
        # print(DataSet[1],'yttttt的数据集')
        # print(dt[0],'显示x的值')
        # print(dt[1],'显示y的值')
        ax.scatter(x=DataSet[0][:,0], y=DataSet[0][:,1], c='k', marker='o',s=10, label='bad')
        ax.scatter(x=DataSet[1][:,0], y=DataSet[1][:,1], c='g', marker='o',s=10, label='good')
        plt.legend(loc='upper right')
        ax.plot(x, z, c='b')
        plt.xlim(0, 1)
        plt.ylim(0, 0.7)
        plt.xlabel('density')
        plt.ylabel('ratio_sugar')
        plt.title('watermelon_3a - LDA')
        k = w[1] / w[0]#注意这个求得的k就是我们要求的分界面的斜率，
        plt.plot([-1, 1], [-k, k])#这个其实是画的(-1,-k)和(1,k)这两点的连线
        for i in range(m):
            curX = (k * DataArr[i, 1] + DataArr[i, 0]) / (1 + k * k)
            if LabelArr[i] == 0:
                plt.plot(curX, k * curX, "ko", markersize=3)
            else:
                plt.plot(curX, k * curX, "go", markersize=3)
            plt.plot([curX, DataArr[i, 0]], [k * curX, DataArr[i, 1]], "c--", linewidth=0.3)
        plt.show()

        # for i in range(len(x0)):
        #     ax.plot([DataArr[:,0][i],DataArr[:,1][i]],[x0[i],y0[i]])
        plt.show()

    elif n==3:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(DataArr[0],DataArr[1],DataArr[2],c='r', marker='o')
        ax.plot_surface(x=x,y=z)
        ax.set_ylabel('X1')
        ax.set_xlabel('X2')
        ax.set_zlabel('X3')
        plt.title('watermelon_3a - LDA')
        plt.show()
    else:
        print('无法显示')


