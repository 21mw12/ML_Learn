# _*_coding:utf-8_*_
import numpy as np
import matplotlib.pyplot as plt
from equation import gradientDescent
from plt_utils import draw_Data, draw_BoundaryFunction


# 数据源
x_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)

# 参数调整
w_init = np.zeros(x_train.shape[1])
b_init = 0
alpha = 0.1
iters = 10000

# 进行逻辑回归
w_final, b_final, J_hist = gradientDescent(x_train, y_train, w_init, b_init, alpha, iters)

# 绘制数据集和边界函数
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
ax.axis([0, 4, 0, 3.5])
draw_Data(ax, x_train, y_train)
draw_BoundaryFunction(ax, w_final, b_final, x_train.max())
plt.show()