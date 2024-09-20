# _*_coding:utf-8_*_
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# 数据源
x_labels = ["AGE", "WORK", "HOME", "LOAN"]
x = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 1, 0, 1],
                    [0, 1, 1, 0],
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [1, 0, 0, 1],
                    [1, 1, 1, 1],
                    [1, 0, 1, 2],
                    [1, 0, 1, 2],
                    [2, 0, 1, 2],
                    [2, 0, 1, 1],
                    [2, 1, 0, 1],
                    [2, 1, 0, 2],
                    [2, 0, 0, 0]])
y = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

# 分类模型
model_c = DecisionTreeClassifier(max_depth=4, max_features=5)
model_c.fit(x_train, y_train)
result_c = model_c.predict(x_test)

# # 回归模型
# model_r = DecisionTreeRegressor(max_depth=4, max_features=5)
# model_r.fit(x_train, y_train)
# result_r = model_r.predict(x_test)

# 结果比对
print(y_test)
print(result_c)


