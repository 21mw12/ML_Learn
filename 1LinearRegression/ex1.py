# _*_coding:utf-8_*_
"""
 单变量线性回归练习1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from equation import gradientDescent
from lab_utils_uni import plt_contour_wgrad, plt_divergence

# 数据源
path = 'data/data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
cols = data.shape[1]    # 获取数据列数
x_train = data.iloc[:, 0:cols - 1].values.transpose()[0]
y_train = data.iloc[:, cols - 1:cols].values.transpose()[0]

# 参数调整
w_init = 0
b_init = 0
alpha = 0.01
iters = 1000

# 进行一次单特征值的机器学习
w_final, b_final, J_hist, p_hist = gradientDescent(x_train, y_train, w_init, b_init, alpha, iters)

x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = b_final + (w_final * x)

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title('Predicted Profit vs. Population Size')
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
plt.show()

# 绘制梯度下降过程以及三维代价函数
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
plt_contour_wgrad(x_train, y_train, p_hist, ax)
plt_divergence(p_hist, J_hist, x_train, y_train)
plt.show()