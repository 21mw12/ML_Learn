# _*_coding:utf-8_*_
import numpy as np

def draw_Data(ax, x, y):
    """
    再图像上绘制数据集，区分两种y
    :param ax: 子图绘制区域
    :param x:  x数据集
    :param y:  y数据集
    """
    pos = y == 1
    neg = y == 0
    pos = pos.reshape(-1, )
    neg = neg.reshape(-1, )

    ax.scatter(x[pos, 0], x[pos, 1], marker='x', s=80, c='red', label="y=1")
    ax.scatter(x[neg, 0], x[neg, 1], marker='o', s=80, c='blue', label="y=0")
    ax.legend(loc='best')

def draw_BoundaryFunction(ax, w, b, x_max):
    """
    在图像上绘制直线的边界函数
    :param ax: 子图绘制区域
    :param w:  参数w
    :param b:  参数b
    :param x_max: 数据集的最大值
    """
    print(x_max)
    x_space = np.linspace(0, x_max, 100)
    y_space = (w[0] * x_space + b) / -w[1]

    ax.plot(x_space, y_space)