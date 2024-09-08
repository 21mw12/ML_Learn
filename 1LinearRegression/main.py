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
    代价函数
    :param x:
    :param y:
    :param theta:
    :return: 代价/成本
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
# print(computeCost(X, Y, theta))

def gradientDescent(X, Y, theta, alpha, iters):
    """
    梯度下降算法
    :param X:
    :param Y:
    :param theta:
    :param alpha: 学习率
    :param iters: 迭代次数
    :return:
    """
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    print(parameters)
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - Y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, Y, theta)

    return theta, cost

alpha = 0.01
iters = 1000

g, cost = gradientDescent(X, Y, theta, alpha, iters)
print(g)

# print(computeCost(X, Y, g))
#
# x = np.linspace(data.Population.min(), data.Population.max(), 100)
# f = g[0, 0] + (g[0, 1] * x)
#
#
#
# fig, ax = plt.subplots(figsize=(12,8))
# ax.plot(x, f, 'r', label='Prediction')
# ax.scatter(data.Population, data.Profit, label='Traning Data')
# ax.legend(loc=2)
# ax.set_xlabel('Population')
# ax.set_ylabel('Profit')
# ax.set_title('Predicted Profit vs. Population Size')
# plt.show()
#
#
# fig, ax = plt.subplots(figsize=(12,8))
# ax.plot(np.arange(iters), cost, 'r')
# ax.set_xlabel('Iterations')
# ax.set_ylabel('Cost')
# ax.set_title('Error vs. Training Epoch')
# plt.show()
#
#
