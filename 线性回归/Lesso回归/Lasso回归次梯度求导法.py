#__author__: dongj
#date: 2018/7/7
import numpy as np
import matplotlib.pyplot as plt

fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(8,8))
plt.suptitle(u'Lasso Shooting Algorithm Example',fontsize = 18) #用中文会出错，不知为何
plt.subplots_adjust(wspace = 0.25,hspace=0.25)
lambds = [0.05,3.5,50,100]
axes = axes.flatten()

for i in range(4):
    lambd = lambds[i]
    numIt = 600 #迭代次数
    wlog = np.zeros((numIt,M)) #记录weights的变化
    weights = np.zeros(M) #系数向量

    XX2 = 2*np.dot(datas.transpose(),datas)
    XY2 = 2*np.dot(datas.transpose(),values)
    for it in range(numIt):
        for k in range(M):
            ck = XY2[k]-np.dot(XX2[k,:],weights)+XX2[k,k]*weights[k]
            ak = XX2[k,k]
            #print ck,lambd
            if ck < -lambd:
                weights[k] = (ck+lambd)/ak
            elif ck > lambd:
                weights[k] = (ck-lambd)/ak
            else:
                weights[k] = 0
        wlog[it,:] = weights[:]
    ax = axes[i]
    for m in range(M):
        ax.plot(wlog[:,m])
    ax.set_title('lambda='+np.str(lambd),{'fontname':'STFangsong','fontsize':10})
    ax.set_xlabel(u'迭代次数',{'fontname':'STFangsong','fontsize':10})
    ax.set_ylabel(u'各权值系数',{'fontname':'STFangsong','fontsize':10})
plt.savefig('lasso2.png',dpi=300,bbox_inches='tight')