# -*- coding: utf-8 -*-
# @Time    : 2018/10/4 16:08
# @Author  : 老湖
# @FileName: figure_plot.py
# @Software: PyCharm
# @qq    ：326883847
import matplotlib.pyplot as plt
# 定义决策树决策结果的属性，用字典来定义
# 下面的字典定义也可写作 decisionNode={boxstyle:'sawtooth',fc:'0.8'}
# boxstyle为文本框的类型，sawtooth是锯齿形，fc是边框线粗细
decisionNode = dict(boxstyle="sawtooth",fc="0.8")
# 定义决策树的叶子结点的描述属性
leafNode = dict(boxstyle="round4",fc="0.8")
# 定义决策树的箭头属性
arrow_args = dict(arrowstyle="<-")

# 绘制结点
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    # annotate是关于一个数据点的文本
    # nodeTxt为要显示的文本，centerPt为文本的中心点，箭头所在的点，parentPt为指向文本的点
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',\
                            va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)
''''' 
# 创建绘图 
def createPlot(): 
    # 类似于Matlab的figure，定义一个画布(暂且这么称呼吧)，背景为白色 
    fig = plt.figure(1,facecolor='white') 
    # 把画布清空 
    fig.clf() 
    # createPlot.ax1为全局变量，绘制图像的句柄，subplot为定义了一个绘图，111表示figure中的图有1行1列，即1个，最后的1代表第一个图 
    # frameon表示是否绘制坐标轴矩形 
    createPlot.ax1 = plt.subplot(111,frameon=False) 
    # 绘制结点 
    plotNode('a decision node',(0.5,0.1),(0.1,0.5),decisionNode) 
    # 绘制结点 
    plotNode('a leaf node',(0.8,0.1),(0.3,0.8),leafNode) 
    plt.show() 
'''
# 获得决策树的叶子结点数目
def getNumLeafs(myTree):
    # 定义叶子结点数目
    numLeafs = 0
    # 获得myTree的第一个键值，即第一个特征，分割的标签
    firstStr = myTree.keys()[0]
    # 根据键值得到对应的值，即根据第一个特征分类的结果
    secondDict = myTree[firstStr]
    # 遍历得到的secondDict
    for key in secondDict.keys():
        # 如果secondDict[key]为一个字典，即决策树结点
        if type(secondDict[key]).__name__ == 'dict':
            # 则递归的计算secondDict中的叶子结点数，并加到numLeafs上
            numLeafs += getNumLeafs(secondDict[key])
        # 如果secondDict[key]为叶子结点
        else:
            # 则将叶子结点数加1
            numLeafs += 1
    # 返回求的叶子结点数目
    return numLeafs

# 获得决策树的深度
def getTreeDepth(myTree):
    # 定义树的深度
    maxDepth = 0
    # 获得myTree的第一个键值，即第一个特征，分割的标签
    firstStr = myTree.keys()[0]
    # 根据键值得到对应的值，即根据第一个特征分类的结果
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 如果secondDict[key]为一个字典
        if type(secondDict[key]).__name__ == 'dict':
            # 则当前树的深度等于1加上secondDict的深度，只有当前点为决策树点深度才会加1
            thisDepth = 1 + getTreeDepth(secondDict[key])
            # 如果secondDict[key]为叶子结点
        else:
            # 则将当前树的深度设为1
            thisDepth = 1
    # 如果当前树的深度比最大数的深度
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    # 返回树的深度
    return maxDepth

# 绘制中间文本
def plotMidText(cntrPt,parentPt,txtString):
    # 求中间点的横坐标
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    # 求中间点的纵坐标
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    # 绘制树结点
    createPlot.ax1.text(xMid,yMid,txtString)

# 绘制决策树
def plotTree(myTree,parentPt,nodeTxt):
    # 定义并获得决策树的叶子结点数
    numLeafs = getNumLeafs(myTree)
    #depth =
    getTreeDepth(myTree)
    # 得到第一个特征
    firstStr = myTree.keys()[0]
    # 计算坐标，x坐标为当前树的叶子结点数目除以整个树的叶子结点数再除以2，y为起点
    cntrPt = (plotTree.xOff + (1.0 +float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    # 绘制中间结点，即决策树结点，也是当前树的根结点，这句话没感觉出有用来，注释掉照样建立决策树，理解浅陋了，理解错了这句话的意思，下面有说明
    plotMidText(cntrPt, parentPt, nodeTxt)
    # 绘制决策树结点
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    # 根据firstStr找到对应的值
    secondDict = myTree[firstStr]
    # 因为进入了下一层，所以y的坐标要变 ，图像坐标是从左上角为原点
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    # 遍历secondDict
    for key in secondDict.keys():
        # 如果secondDict[key]为一棵子决策树，即字典
        if type(secondDict[key]).__name__ == 'dict':
            # 递归的绘制决策树
            plotTree(secondDict[key],cntrPt,str(key))
        # 若secondDict[key]为叶子结点
        else:
            # 计算叶子结点的横坐标
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            # 绘制叶子结点
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt, leafNode)
            # 这句注释掉也不影响决策树的绘制,自己理解的浅陋了，这行代码是特征的值
            plotMidText((plotTree.xOff,plotTree.yOff), cntrPt, str(key))
    # 计算纵坐标
    plotTree.yOff = plotTree.yOff +1.0/plotTree.totalD

def createPlot(inTree):
    # 定义一块画布(画布是自己的理解)
    fig = plt.figure(1,facecolor='white')
    # 清空画布
    fig.clf()
    # 定义横纵坐标轴，无内容
    axprops = dict(xticks=[],yticks=[])
    # 绘制图像，无边框，无坐标轴
    createPlot.ax1 = plt.subplot(111,frameon=True,**axprops)
    # plotTree.totalW保存的是树的宽
    plotTree.totalW = float(getNumLeafs(inTree))
    # plotTree.totalD保存的是树的高
    plotTree.totalD = float(getTreeDepth(inTree))
    # 决策树起始横坐标
    plotTree.xOff = - 0.5 / plotTree.totalW #从0开始会偏右
    print(plotTree.xOff)
    # 决策树的起始纵坐标
    plotTree.yOff = 1.0
    # 绘制决策树
    plotTree(inTree,(0.5,1.0),'')
    # 显示图像
    plt.show()

# 预定义的树，用来测试
def retrieveTree(i):
    listOfTree = [{'no surfacing':{ 0:'no',1:{'flippers': \
                       {0:'no',1:'yes'}}}},
                   {'no surfacing':{ 0:'no',1:{'flippers': \
                    {0:{'head':{0:'no',1:'yes'}},1:'no'}}}}
                  ]
    return listOfTree[i]