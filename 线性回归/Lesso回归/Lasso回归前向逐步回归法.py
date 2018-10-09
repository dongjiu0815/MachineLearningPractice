#__author__: dongj
#date: 2018/7/7
# import numpy as np
# import matplotlib.pyplot as plt
# from copy import copy as cp
# datas = []
# values = []
# with open('abalone','r') as f:
#     for line in f:
#         linedata =  line.split('\t')
#         datas.append(linedata[0:-1]) #前4列是4个属性的值
#         values.append(linedata[-1].replace('\n',''))  #最后一列是类别
# datas = np.array(datas)
# datas = datas.astype(float)
# values = np.array(values)
# values = values.astype(float)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import copy as cp
from sklearn import datasets,linear_model
datas=pd.read_csv('Folds5x2_pp.csv')
X = datas[['AT', 'V', 'AP', 'RH']]
y = datas[['PE']]
N,M =  datas.shape #N是样本数，M是参数向量的维
means = datas.mean(axis=0) #各个属性的均值
stds = datas.std(axis=0) #各个属性的标准差
datas = (datas-means)/stds #标准差归一化
values = (y-y.mean())/y.std() #标准差归一化


fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(8,8))
plt.suptitle(u'Forward Stepwise Regression Example',fontsize = 18) #用中文会出错，不知为何
plt.subplots_adjust(wspace = 0.25,hspace=0.25)
lambds = [0.05,0.5,1.0,3.0]

axes = axes.flatten()
for i in range(4):
    numIt = 600 #迭代次数
    delta = 0.01 # 调整系数
    wlog = np.zeros((numIt,M)) #记录weights的变化
    weights = np.zeros(M) #系数向量
    lambd = lambds[i]

    for it in range(1,numIt):
        Lmin = {'value':np.inf,'loc':np.nan,'sign':np.nan} #记录本次迭代的目标函数最小值
        for m in range(M-1,0,-1):
            for sign in (-1,1):
                wbak = cp(weights)
                wbak[m] += delta*sign
                Lcur = np.linalg.norm(values-np.dot(datas,wbak),2)+ lambd*np.linalg.norm(wbak,1)
                #print m,sign,Lcur
                if Lmin['value'] > Lcur: # 如果目标函数值比当前最优值小
                    Lmin['value'] = Lcur
                    Lmin['loc'] = m
                    Lmin['sign'] = sign
        weights[Lmin['loc']] += delta*Lmin['sign']
        wlog[it,:] = weights[:]
    ax = axes[i]
    for m in range(M):
        ax.plot(wlog[:,m])
    ax.set_title('lambda='+np.str(lambd),{'fontname':'STFangsong','fontsize':10})
    ax.set_xlabel(u'迭代次数',{'fontname':'STFangsong','fontsize':10})
    ax.set_ylabel(u'各权值系数',{'fontname':'STFangsong','fontsize':10})
plt.savefig('lasso1.png',dpi=300,bbox_inches='tight')