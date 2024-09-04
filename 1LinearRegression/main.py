import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
# print(data.head())  # 默认读取前5行数据用于测试
# print(data.describe())  # 统计数据（最大值，最小值，总数....）

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
# plt.show()  # 用数据绘制散点图


def computeCost(x, y, theta):
    """
    成本函数（代价函数）
    """
    inner = np.power(((x * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(x))


# 向数据中插入一列（插入第0列，数据为1，取名为Ones）
data.insert(0, 'Ones', 1)

# 提取数据
cols = data.shape[1]    # 获取数据列数
X = data.iloc[:, 0:cols - 1]  # 获取所有行，除了最后一列
Y = data.iloc[:, cols - 1:cols]  # 获取所有行，只有最后一列
# print(X.head())
# print(Y.head())

# 转换为矩阵
X = np.matrix(X.values)
Y = np.matrix(Y.values)
theta = np.matrix(np.array([0, 0]))

# 计算成本
print(computeCost(X, Y, theta))