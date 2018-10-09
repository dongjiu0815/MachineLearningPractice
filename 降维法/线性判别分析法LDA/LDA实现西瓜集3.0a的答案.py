# -*- coding: utf-8 -*-
# @Time    : 2018/8/29 20:43
# @Author  : 老湖
# @FileName: LDA实现西瓜集3.0a的答案.py
# @Software: PyCharm
# @qq    ：326883847
import numpy as np  # for matrix calculation

# load the CSV file as a numpy matrix
# separate the data with " "(blank,\t)
dataset = np.loadtxt('watermelon3_0a.csv', delimiter=",")
print(dataset)
# separate the data from the target attributes
X = dataset[:, 1:3]
y = dataset[:, 3]
goodData = dataset[:8]
badData = dataset[8:]
# return the size
m, n = np.shape(X)
# print(m,n)#17,2
# draw scatter diagram to show the raw data
# https://matplotlib.org/api/pyplot_summary.html

'''
LDA via sklearn
'''
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
import matplotlib.pyplot as plt

# generalization of train and test set
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=0)
# model fitting
# http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.
# LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis
lda_model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None).fit(X_train, y_train)
# model validation
y_pred = lda_model.predict(X_test)
# summarize the fit of the model
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

f1 = plt.figure(1)
plt.title('watermelon_3a')
plt.xlabel('density')
plt.ylabel('ratio_sugar')
"""
plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='b', s=100, label='bad')
"""
plt.scatter(goodData[:, 1], goodData[:, 2], marker='o', color='g', s=100, label='good')
plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='k', s=100, label='bad')
plt.legend(loc='upper right')

'''
implementation of LDA based on self-coding
'''
u = [[badData[:, 1].mean(), badData[:, 2].mean()], [goodData[:, 1].mean(), goodData[:, 2].mean()]]
u = np.matrix(u)

Sw = np.zeros((n, n))
for i in range(m):
    x_tmp = X[i].reshape(n, 1)  # row -> cloumn vector
    if y[i] == 0: u_tmp = u[0].reshape(n, 1)
    if y[i] == 1: u_tmp = u[1].reshape(n, 1)
    Sw += np.dot(x_tmp - u_tmp, (x_tmp - u_tmp).T)
print(Sw,'展示Sw')
U, sigma, V = np.linalg.svd(Sw)
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.diag.html
Sw_inv = V.T * np.linalg.inv(np.diag(sigma)) * U.T
print(Sw_inv,'Sw_inv的值')
# print(np.linalg.inv(Sw))
# 3-th. computing the parameter w, refer on book (3.39)
w = np.dot(Sw_inv, (u[0] - u[1]).reshape(n, 1))  # here we use a**-1 to get the inverse of a ndarray
print(w,'w权值的大小')
# P=[]
# for i in range(2): # two class
#     P.append(np.mean(X[y==i], axis=0)) # column mean

f3 = plt.figure(3)
plt.xlim(0, 1)
plt.ylim(0, 0.7)

plt.title('watermelon_3a - LDA')
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='k', s=10, label='bad')
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='g', s=10, label='good')
plt.legend(loc='upper right')

k = w[1, 0] / w[0, 0]
plt.plot([-1, 1], [-k, k])

for i in range(m):
    curX = (k * X[i, 1] + X[i, 0]) / (1 + k * k)
    if y[i] == 0:
        plt.plot(curX, k * curX, "ko", markersize=3)
    else:
        plt.plot(curX, k * curX, "go", markersize=3)
    plt.plot([curX, X[i, 0]], [k * curX, X[i, 1]], "c--", linewidth=0.3)

plt.show()
