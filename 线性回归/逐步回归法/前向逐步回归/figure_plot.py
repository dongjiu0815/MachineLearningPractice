#__author__: dongj
#date: 2018/8/4
def plotBestFit(WeightSet):
    import matplotlib.pyplot as plt
    import numpy as np
    fig=plt.figure()
    ax=fig.add_subplot(111)
    m=np.shape(WeightSet)[0]
    x=list(range(m))
    y=WeightSet
    plt.ylim(-1.0,1.5)
    ax.plot(x,y,'-')
    plt.show()