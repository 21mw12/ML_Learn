"""
 此文件用于编写关于这章的相关公式/算法
"""
import math

def computeCost(x, y, w, b):
    """
    计算代价函数计算代价函数
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

    """
    梯度下降函数
     Args:
      x    : 每个数据点的x坐标
      y    : 每个数据点的y坐标
      w    : 模型参数w
      b    : 模型参数b，截距
    Returns
      cost : 由参数w和b计算出的代价
    """
    # 记录数据用于绘图
    J_history = []
    p_history = []
    for i in range(iters):
        dj_dw, dj_db = computeGradient(x, y, w, b)
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        # 保存历史数据，避免资源耗尽，多余的部分将打印在控制台
        if i < 100000:
            J_history.append(computeCost(x, y, w, b))
            p_history.append([w, b])
        if i % math.ceil(iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w, b, J_history, p_history