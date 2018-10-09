#__author__: dongj
#date: 2018/7/16
def plotBestFit(weight):
    import matplotlib.pyplot as plt
    fig=plt.figure()
    ax=fig.add_subplot(111)
    x=[(i-10) for i in range(30)]
    y=weight
    #print(x.shape)
    print(y.shape)
    ax.plot(x,y)
    plt.show()