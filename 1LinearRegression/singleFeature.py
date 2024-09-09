# _*_coding:utf-8_*_
import numpy as np
import matplotlib.pyplot as plt
from equation import gradientDescent
from lab_utils_uni import plt_contour_wgrad, plt_divergence

# 数据源
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

# 参数调整
w_init = 0
b_init = 0
alpha = 1.0e-2
iters = 10000

# 进行单特征值的机器学习
w_final, b_final, J_hist, p_hist = gradientDescent(x_train, y_train, w_init, b_init, alpha, iters)

# 绘制前100次迭代之后的成本和1000次之后的迭代之后的成本
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax1.set_title("Cost vs. iteration(start)")
ax1.set_ylabel('Cost')
ax1.set_xlabel('iteration step')
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax2.set_title("Cost vs. iteration (end)")
ax2.set_ylabel('Cost')
ax2.set_xlabel('iteration step')
plt.show()

# 绘制梯度下降过程以及三维代价函数
# fig, ax = plt.subplots(1, 1, figsize=(12, 6))
# plt_contour_wgrad(x_train, y_train, p_hist, ax)
# plt_divergence(p_hist, J_hist, x_train, y_train)
# plt.show()