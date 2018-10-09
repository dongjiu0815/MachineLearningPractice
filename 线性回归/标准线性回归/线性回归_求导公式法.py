#__author__: dongj
#date: 2018/7/17
#数据加载方式一
# def loadDataSet(fileName):
#     datasize=len(open(fileName).readline().split('\t'))-1
#     #print(open(fileName).readline().split('\t'))
#     fr=open(fileName)
#     dataArr=[]
#     labelArr=[]
#     #print(datasize)
#     for line in fr.readlines():
#         data=[]
#         #print(line,'ggg')
#         curLine=list(line.strip().split('\t'))
#         #print(curLine,'fff')
#         if datasize==1:
#             data.append(float(curLine[0]))
#         else:
#             for i in range(datasize):
#                 print(curLine[i])
#                 data.append(float(curLine[i]))
#         #print(data,'ppp')
#         dataArr.append(data)
#         #print(curLine[-1],'hhh')
#         labelArr.append(float(curLine[-1]))
#     return dataArr,labelArr


#数据加载方式三
# def loadDataSet(filename):
#     import numpy as np
#     dataArr,labelArr=[],[]
#     with open(filename,'r') as txtfile:
#         for line in txtfile.readlines():
#             dataArr.append([float(i) for i in line.strip().split('\t')][:-1])
#             labelArr.append(list(map(float,line.strip().split('\t')))[-1])
#     return dataArr,labelArr
#数据加载方式二


def loadDataSet(filename):
    import numpy as np
    dataArr,labelArr,cur=[],[],[]
    with open(filename) as textfile:
        for line in textfile.readlines():
            cur.append(line.strip().split('\t'))
    cur=np.array(cur,dtype='float32')
    np.random.shuffle(cur)#将样本的顺序打乱
    dataArr,labelArr=cur[:,:-1],cur[:,-1]
    return dataArr,labelArr

def standRegRes(dataArr,labelArr):
    import numpy as np
    dataArr=np.mat(dataArr)
    labelArr=np.array(labelArr).reshape(-1,1)
    m = np.shape(dataArr)[0]
    # labelArr.shape=(-1,1)
    # labelArr=labelArr.transpose()
    # print(dataArr,'ggg')
    # print(labelArr,'ddd')
    # print(np.linalg.det(dataArr.T*dataArr),'hhh')
    if np.linalg.det(dataArr.T*dataArr)==0.0:
        print('this matrix is singular,cannot do inverse')
    else:
        # print(dataArr[:,0],'qqq')
        # print(len(dataArr)*1,'www')
        # print(dataArr[:,0]==len(dataArr)*1,'eee')
        #print(type(dataArr[:,0]))
        #print(type(dataArr.shape[0]*[1]))
        if (dataArr[:,0].all()==np.shape(dataArr)[0]*[1]).all():
            #print(np.shape(dataArr),np.shape(labelArr))
            return np.dot(np.linalg.inv(dataArr.T*dataArr),dataArr.T*labelArr)
        else:
            np.hstack((np.array(m*[1]).reshape(m,1),dataArr))
            return np.dot(np.linalg.inv(dataArr.T*dataArr),dataArr.T*labelArr)
