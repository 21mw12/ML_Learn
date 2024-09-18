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

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

model_c = DecisionTreeClassifier(max_depth=4, max_features=5)
# 训练分类模型
model_c.fit(x_train, y_train)
# 使用模型预测分类结果
result_c = model_c.predict(x_test)

print(y_test)
print(result_c)

# model_r = DecisionTreeRegressor(max_depth=4, max_features=5)
# # 训练回归模型
# model_r.fit(x_train, y_train)
# result_r = model_r.predict(x_test)
