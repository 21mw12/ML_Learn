# _*_coding:utf-8_*_
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from plt_utils import draw_BoundaryFunction, draw_Data

# 数据源
path = 'data/data1.txt'
data = pd.read_csv(path, header=None)
m, n = data.shape    # 获取数据列数
x_train = np.array(data.iloc[:, 0:n-1])
y_train = np.array(data.iloc[:, n - 1:n]).reshape(1, m)[0]

# 实例化一未训练的线性模型
model = LogisticRegression()

# 调用评估器对数据进行训练
model = model.fit(x_train, y_train)

w_final = model.coef_[0]
b_final = model.intercept_
print(w_final, b_final)

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title('Predicted Profit vs. Population Size')
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
draw_Data(ax, x_train, y_train)
draw_BoundaryFunction(ax, w_final, b_final, x_train.max())
ax.legend(loc=2)
plt.show()
