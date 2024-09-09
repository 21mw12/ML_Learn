"""
 此文件用于编写关于这章的相关公式/算法
"""
import copy
import math
import numpy as np

'''
 单特征值会用到的函数
'''
def computeCost(x, y, w, b):
    """
    计算代价函数
     Args:
      x : 每个数据点的x坐标
      y : 每个数据点的y坐标
      w : 模型参数w
      b : 模型参数b，截距
    Returns:
      cost : 由参数w和b计算出的代价
    """
    # 得到数据总量
    m = x.shape[0]
    # 由w和b计算出来的代价
    cost = 0

    # 遍历所有数据
    for i in range(x.shape[0]):
        # 计算函数值
        f_wb = w * x[i] + b
        # 累计误差的方差
        cost = cost + (f_wb - y[i]) ** 2
    # 计算代价
    cost = 1 / (2 * m) * cost

    return cost

def computeGradient(x, y, w, b):
    """
    计算代价函数的两个偏导数结果，即梯度
     Args:
      x : 每个数据点的x坐标
      y : 每个数据点的y坐标
      w : 模型参数w
      b : 模型参数b，截距
    Returns:
      dj_dw : 模型参数w的下降梯度
      dj_db : 模型参数b的下降梯度
    """
    # 得到数据量
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    # 遍历所有数据
    for i in range(m):
        # 计算函数值
        f_wb = w * x[i] + b
        # 中间值
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
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
      p_history : 模型参数w和b的历史结果（用于绘图）
    """
    # 记录数据用于绘图
    J_history = []
    p_history = []
    for i in range(iters):
        dj_dw, dj_db = computeGradient(x, y, w, b)
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        # 保存历史数据
        if i < 100000:
            J_history.append(computeCost(x, y, w, b))
            p_history.append([w, b])
        # 打印训练过程信息
        if i % math.ceil(iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e}")
    return w, b, J_history, p_history

'''
 多特征值会用到的函数
'''
def computeMVFunction(x, w, b):
    """
    计算多特征值的函数模型值
     Args:
      x : 每个数据点的x坐标
      w : 模型参数w
      b : 模型参数b，截距
    Returns:
      f_wnb : 函数模型值
    """
    f_wnb = np.dot(x, w) + b
    return f_wnb

def computeMVCost(x, y, w, b):
    """
    计算多特征值的代价函数
     Args:
      x : 每个数据点的x坐标
      y : 每个数据点的y坐标
      w : 模型参数w
      b : 模型参数b，截距
    Returns:
      cost : 由参数w和b计算出的代价
    """
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wnb_i = computeMVFunction(x[i], w, b)
        cost = cost + (f_wnb_i - y[i]) ** 2
    cost = cost / (2 * m)  # scalar
    return cost

def computeMVGradient(x, y, w, b):
    """
    计算多特征值的代价函数的两个偏导数结果，即梯度
     Args:
      x : 每个数据点的x坐标
      y : 每个数据点的y坐标
      w : 模型参数w
      b : 模型参数b，截距
    Returns:
      dj_dw : 模型参数w的下降梯度
      dj_db : 模型参数b的下降梯度
    """
    m, n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(x[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * x[i, j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradientMVDescent(x, y, w, b, alpha, iters):
    """
    多特征值的梯度下降函数
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
      p_history : 模型参数w和b的历史结果（用于绘图）
    """
    J_history = []

    for i in range(iters):
        dj_dw, dj_db = computeMVGradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_history.append(computeMVCost(x, y, w, b))
        if i % math.ceil(iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

    return w, b, J_history

