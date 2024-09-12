# _*_coding:utf-8_*_
"""
 逻辑回归练习1
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from equation import gradientDescent
from plt_utils import draw_Data, draw_BoundaryFunction

# 数据源
path = 'data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
m, cols = data.shape    # 获取数据列数
x_train = np.array(data.iloc[:, 0:cols-1].values)
y_train = np.array(data.iloc[:, cols-1:cols].values).transpose()[0]

# 参数调整（运行时间过长）
w_init = np.zeros(x_train.shape[1])
b_init = 0
alpha = 0.001
iters = 200000

# 进行逻辑回归
w_final, b_final, J_hist = gradientDescent(x_train, y_train, w_init, b_init, alpha, iters)
print(w_final, b_final)

# 绘制迭代过程中的成本
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(J_hist)
ax.set_title("cost vs. iteration")
ax.set_ylabel('Cost')
ax.set_xlabel('iteration step')
plt.show()

# 绘制数据集和边界函数
fig, ax = plt.subplots(figsize=(12,8))
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
draw_Data(ax, x_train, y_train)
draw_BoundaryFunction(ax, w_final, b_final, x_train[:, 0].max())
plt.show()