#__author__: dongj
#date: 2018/5/14

import numpy as np
import time

#loadData,ndarray类型
def loadData(fileName):
    train_x = [];train_y = []
    with open(fileName) as fr:
        lineArr=[]
        for line in fr.readlines():
            lineCur = line.strip().split('\t')
            lineArr.append(lineCur)
    lineArr=np.array(lineArr,dtype='float')
    DataArr,LabelArr=lineArr[:,:-1],lineArr[:,-1]
    return DataArr,LabelArr

# calculate the rssError
def rssError(y_pre,yArr):
    return sum((y_pre-yArr)**2)

#标准线性回归
# train a logistic regression model using some optional optimize algorithm
# input: xTrain is a mat datatype, each row stands for one sample
#        yTrain is mat datatype too, each row is the corresponding label
#        opts is optimize option include step and maximum number of iterations
def StandardRegres(DataArr, LabelArr, opts):
    # calculate training time
    import numpy as np
    startTime = time.time()
    m, n= np.shape(DataArr)
    alpha = opts['alpha']
    IterMax = opts['IterMax']
    eps=opts['eps']
    b=opts['b']
    weights = np.ones((n, 1))
    count=0.0

    # optimize through gradient descent algorilthm
    IterMaxNum=0
    for k in range(IterMax):
        if opts['optimizeType']== 'DerivationFormula':
            count+=1
            DataArr=np.mat(DataArr)
            if np.linalg.det(DataArr.T * DataArr) == 0.0:
                print('this matrix is singular,cannot do inverse')
            else:
                weights=np.dot((DataArr.T * DataArr).I, DataArr.T * LabelArr.reshape(-1,1))
                weights=np.array(weights)
                #print(type(weights),'展示weight的类型')
        elif opts['optimizeType'] == 'BatchGradDescent':  #Batch gradient descent algorilthm
            count += 1
            error = np.dot(DataArr, weights) - LabelArr
            weights-=alpha * np.dot(DataArr.T, error)
        elif opts['optimizeType'] == 'StocGradDescent':  # Stochastic gradient descent
            for i in range(m):
                count += 1
                error = np.dot(DataArr[i, :], weights) - LabelArr[i]
                # print(weight.shape,'aaa')
                # print(dataArr[i].T.shape,'nnn')
                weights -= alpha * error[0, 0] * DataArr[i, :].T
                if np.linalg.norm(error, ord=np.inf) <= eps:
                    break
        elif opts['optimizeType'] == 'smoothStocGradDescent':  # smooth stochastic gradient descent
            # randomly select samples to optimize for reducing cycle fluctuations
            import numpy as np
            count = 0.0
            weights=weights.flatten()
            # print(weights.shape,'权重')
            # print(type(weights),'权重')
            for i in range(m):
                count += 1
                alpha = 4.0 / (i + k + 1.0) + 0.01  # 第一处改进
                randomindex = int(np.random.uniform(0, m))  # 第二处改进
                #print(randomindex,'aaa')
                # print(DataArr.shape)
                # print(type(DataArr))
                # print(weights.shape)
                # print(type(weights))
                yMat=DataArr[randomindex] * weights.reshape(-1,1)
                # print(type(yMat),'想看的类型',i)
                # print(yMat.shape,'想看的维度',i)
                h=np.sum(yMat)
                #print(h, 'h的值',i)
                #print(h.shape,'h的维度')
                #print(LabelArr,'ccc')
                error = LabelArr[randomindex] - h
                #print(error,'aaaa')
                DataArr = np.array(DataArr)  # 由于dataMatrix是一个列表所以应该将变成跟其他一样的ndarray型
                # print(type(weights),'fdf')
                # print(weights.shape,'fdsfdsg')
                # print(type(weights),'tyyy')
                # print(error,'fdsa')
                # print(randomindex,'fjjj')
                # print(DataArr[randomindex].shape,'fshjty')
                # print(type(DataArr[randomindex]),'poo')
                #weights=weights.tolist()
                # print(weights.shape,'guieahg')
                # print(type(weights), 'ojoo')
                # print(error.shape,'error')
                # print(error)
                weights =weights+alpha * error * (DataArr[randomindex])
                # print(DataArr[randomindex].shape, 'fshjty__')
                # print(type(DataArr[randomindex]), 'poo__')
                # print(weights.shape, 'guieahg__')
                # print(type(weights), 'ojoo__')
                np.delete(DataArr, randomindex, 0)# during one interation, delete the optimized sample
                #print(DataArr,'删除后的数据')
        elif opts['optimizeType'] == 'MiniBatchGradDescent':
            for j in range(n):
                count += 1
                error = np.dot(DataArr, weights) - LabelArr[:, np.newaxis]
                print(error.shape)
                SumNum = 0
                if isinstance(b, (int)):
                    if b < m:
                        pass
                    else:
                        print('超出了样本数,请重新输入')
                elif b.split('.') == 0:
                    b = b * m
                else:
                    print('请重新输入b,b要么是0到1之间的小数,要么是小于样本数的整数')

                for i in range(j * b, b * j + b):
                    count += 1
                    SumNum += DataArr[i, j] * error[i, 0]
                weights -= alpha * SumNum
                if np.linalg.norm(error, ord=np.inf) < eps:
                    break

        else:
            raise NameError('Not support optimize method type!')

        #print('the',k, 'training complete！' )
    IterMaxNum += 1
    #print('Congratulations the', IterMaxNum, '500 times training complete! Took %fs!' % (time.time() - startTime))
    weights=np.array(weights)
    return weights

#均方误差MSE
def rssErrorMse(y_pre,yArr):
    return sum((y_pre-yArr)**2)
#模型拟合优度R^2
def rssError(y_pre,yArr):
    import numpy as np
    yArrMean = np.mean(yArr)
    SSR=np.sum((y_pre-yArrMean)**2)
    #print(SSR,'ssr的值')
    SST=np.sum((yArr-yArrMean)**2)
    #print(SST,'sst的值')
    return SSR/SST

# 前向逐步选择法
def ForwardStepwise(DataArr, LabelArr, opts):
    DataArr=np.array(DataArr)
    # print(DataArr.shape,'原始自变量数据的维度')
    # print(LabelArr.shape,'原始应变量数据的维度')
    #print(type(DataArr),'原始数据的类型')
    m,n=np.shape(DataArr)
    Model={i:i for i in range(n-1)}
    #print(m,n)
    for k in range(n-1):
        from itertools import combinations
        combins=[c for c in combinations(range(n-1),k+1)]
        rss={}
        #print(combins,'查看combins是多少')
        for i in combins:
            #print(i,'i的值是')
            DataRemov=DataArr[:,1:]
            #LabelRemov=LabelArr[1:-1]
            #print(LabelRemov,'remove')
            # print(i,'i的数值')
            # print(DataRemov.shape,'取出第一列后的数据的维度')
            DataMat=DataRemov[:,list(i)]
            #LabelMat=LabelRemov[list(i)]
            #print(LabelMat,'aaa')
            FirstCol=DataArr[:,0].reshape(len(LabelArr),1)
            DataMat=np.hstack((FirstCol,DataMat))
            #print(DataMat,'得到的数据集是')
            #LabelMat=list(LabelMat)
            #LabelMat=LabelMat.insert(0,LabelArr[0])
            #print(DataMat,'aaa')
            #print(LabelMat,'bbb')
            #print(LabelMat)
            LabelMat=LabelArr
            weight=StandardRegres(DataMat, LabelMat, opts)
            y_pre=np.dot(DataMat,weight.reshape(-1,1))
            y_pre=y_pre.flatten()
            # print(y_pre,'得到的预测值')
            # print(LabelArr,'得到的实际值')
            # print(y_pre.shape,'预测值的维度')
            # print(LabelMat.shape,'实际值的维度')
            # print(type(rssError(y_pre,LabelMat)),'误差的类型')
            # print(y_pre,'展示y_pre的值')
            # print(LabelMat,'展示LabelMat的值')
            # print(rssError(y_pre,LabelMat),'显示误差值')
            rss[rssError(y_pre,LabelMat)]=[i,DataMat,LabelMat,y_pre,rssError(y_pre,LabelMat)]
        rss_all=rss.keys()
        #print(rss_all,'所有的r^2的值')
        rss_max=max(rss_all)
        Model[k]=rss[rss_max]
    return Model

#the model select Method
#赤池信息准则
def AIC_Score(y_pre,yArr,m,n):
    import numpy as np
    SSR = sum((y_pre - yArr) ** 2)
    return 2*n+m*np.log(SSR/m)
#贝叶斯信息准则
def BIC_Score(y_pre,yArr,m,n):
    import numpy as np
    SSR=sum((y_pre-yArr)**2)
    sigma=np.std(y_pre-yArr)
    return (SSR+n*sigma*np.log(m))/m

#调整决定系数R^2
def R2_Score(y_pre,yArr,m,n):
    import numpy as np
    SSR=sum((y_pre-yArr)**2)
    SST=sum((yArr-np.mean(yArr,axis=0))**2)
    return 1-((SSR/(m-n-1))/(SST/(m-1)))
#cp统计量
def CP_Score(y_pre,yArr,m,n):
    import numpy as np
    SSR = sum((y_pre - yArr) ** 2)
    sigma = np.std(y_pre - yArr)
    return (SSR+2*n*sigma)/m

#model select
def ModelSelect(M,Method='CP_Score'):
    #print(len(M))
    Score = []
    #count=0
    for j in M.values():
        if Method=='CP_Score':
            # print(j,'j的值')
            # print(type(j),'j的类型')
            DataArr=j[1]
            # print(DataArr,'展示数据集')
            # print(type(DataArr), '展示数据集的类型')
            m,n=np.shape(DataArr)
            #print(m,n,DataArr)
            # print(j[3].shape,'预测值的维度')
            # print(j[2].shape,'实际值的维度')
            #count+=1
            #print('这是第',count,'次模型得分的计算,其得分为',CP_Score(j[3].flatten(),j[2],m,n))
            Score.append(CP_Score(j[3],j[2],m,n))
        elif Method=='BIC_Score':
            DataArr = j[1]
            m, n = np.shape(DataArr)
            Score.append(BIC_Score(j[3], j[2], m, n))
        elif Method=='AIC_Score':
            DataArr = j[1]
            m, n = np.shape(DataArr)
            Score.append(AIC_Score(j[3], j[2], m, n))
        elif Method=='R2_Score':
            DataArr = j[1]
            m, n = np.shape(DataArr)
            Score.append(AIC_Score(j[3], j[2], m, n))
    print(Score,'展示n-1个模型的的分值')
    print(type(Score),'展示模型得分的类型')
    Score_max=max(Score)
    print(Score_max,'Score的最大值')
    print(Score.index(Score_max),'展示模型最好的那个')
    return M[Score.index(Score_max)]


















