# _*_coding:utf-8_*_
import math
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def computeCost(x, y, w, b):
    m, n = x.shape
    cost = 0.0

    for i in range(m):
        f_wb = sigmoid(np.dot(x[i], w) + b)
        cost += -y[i] * np.log(f_wb + 1e-5) - (1 - y[i]) * np.log(1 - f_wb + 1e-5)
    cost = cost / m
    return cost

def computeGradient(x, y, w, b):
    # 得到数据量
    m, n = x.shape
    dj_dw = np.zeros(n)
    dj_db = 0

    # 遍历所有数据
    for i in range(m):
        # 计算函数值
        f_wb = sigmoid(np.dot(x[i], w) + b)
        err_i = f_wb - y[i]
        for j in range(n):
            dj_dw[j] += err_i * x[i, j]
        dj_db += err_i

    # 得到最终的代价函数的两个偏导数结果
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradientDescent(x, y, w, b, alpha, iters):
    """
    梯度下降函数
     Args:
      x     : 每个数据点的x坐标
      y     : 每个数据点的y坐标
      w     : 模型参数w
      b     : 模型参数b，截距
      alpha : 学习率
      iters : 迭代次数
    Return:
      w         : 最终计算出的参数w
      b         : 最终计算出的参数w
      J_history : 代价函数J_wb的历史结果（用于绘图）
    """
    # 记录数据用于绘图
    J_history = []

    for i in range(iters):
        dj_dw, dj_db = computeGradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # 保存历史数据
        if i < 100000:
            J_history.append(computeCost(x, y, w, b))
        # 打印训练过程信息
        if i % math.ceil(iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
    return w, b, J_history
