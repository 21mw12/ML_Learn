# _*_coding:utf-8_*_
import math
import numpy as np
import matplotlib.pyplot as plt

# 数据源
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

def computeCost(x, y, w, b):
    """
    计算代价函数
    :param x: 每个数据点的x坐标
    :param y: 每个数据点的y坐标
    :param w: 参数w
    :param b: 参数b，截距
    :return: 代价
    """
    # 得到数据量
    m = x.shape[0]
    cost = 0

    # 遍历所有数据
    for i in range(x.shape[0]):
        # 计算函数值
        f_wb = w * x[i] + b
        # 累计误差的方差
        cost = cost + (f_wb - y[i]) ** 2
    # 计算代价
    total_cost = 1 / (2 * m) * cost

    return total_cost

def computeGradient(x, y, w, b):
    """
    计算代价函数的两个偏导数结果，即梯度
    :param x: 每个数据点的x坐标
    :param y: 每个数据点的y坐标
    :param w: 参数w
    :param b: 参数b，截距
    :return: 两个参数的下降梯度
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
    :param x: 每个数据点的x坐标
    :param y: 每个数据点的y坐标
    :param w: 参数w
    :param b: 参数b，截距
    :param alpha: 学习率
    :param iters: 迭代次数
    :return:
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

# 参数调整
w_init = 0
b_init = 0
alpha = 1.0e-2
iters = 10000

# 进行一次单特征值的机器学习
w_final, b_final, J_hist, p_hist = gradientDescent(x_train, y_train, w_init, b_init, alpha, iters)

# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("前100次迭代的成本");  ax2.set_title("1000次之后的迭代成本")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step')
# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.show()
