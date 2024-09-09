# _*_coding:utf-8_*_
import numpy as np
import matplotlib.pyplot as plt
from equation import gradientMVDescent

# 设置只显示小数点后两位
np.set_printoptions(precision=2)

# 数据源
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# 参数调整
w_init = np.zeros_like(X_train[0,:])
b_init = 0.
alpha = 5.0e-7
iters = 1000

# 进行多特征值的机器学习
w_final, b_final, J_hist = gradientMVDescent(X_train, y_train, w_init, b_init, alpha, iters)

# 绘制迭代过程中的成本变化
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax1.set_title("Cost vs. iteration")
ax1.set_ylabel('Cost')
ax1.set_xlabel('iteration step')
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax2.set_title("Cost vs. iteration (tail)")
ax2.set_ylabel('Cost')
ax2.set_xlabel('iteration step')
plt.show()