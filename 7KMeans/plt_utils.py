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
