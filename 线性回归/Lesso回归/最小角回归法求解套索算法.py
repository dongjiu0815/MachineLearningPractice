#__author__: dongj
#date: 2018/7/7
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn import datasets

# 导入数据集
# 这个数据集，总的样本个数为442个，特征维度为10
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
print(X.shape)

# 所谓参数正则化路径
# 其实就是LARS算法每次迭代的时候，每个参数的数值所组成的曲线
# 其横轴对应着迭代的程度，纵轴是每个特征参数对应的数值
# 这里一共有10个特征，所以有10条特征正则化曲线
print("基于LARS算法计算正则化路径：")
alphas, _, coefs = linear_model.lars_path(X, y, method='lasso', verbose=True)

# 这里讲迭代程度归一化到[0,1]直间
xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')
plt.show()