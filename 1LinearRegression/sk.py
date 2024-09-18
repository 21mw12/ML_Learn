# _*_coding:utf-8_*_
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 数据源
path = 'data1.txt'
data = pd.read_csv(path, header=None)
m, n = data.shape    # 获取数据列数
x_train = np.array(data.iloc[:, 0:n-1])
y_train = np.array(data.iloc[:, n - 1:n])

# 实例化一未训练的线性模型
model = LinearRegression()

# 调用评估器对数据进行训练
model.fit(x_train, y_train)

print(mean_squared_error(model.predict(x_train), y_train))

w_final = model.coef_[0]
b_final = model.intercept_
print(w_final, b_final)

x = np.linspace(x_train.min(), x_train.max(), 100)
f = b_final + (w_final * x)

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title('Predicted Profit vs. Population Size')
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.plot(x, f, 'r')
ax.scatter(x_train, y_train, label='Traning Data')
ax.legend(loc=2)
plt.show()