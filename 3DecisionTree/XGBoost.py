# _*_coding:utf-8_*_
import numpy as np
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# 数据源
# 该示例的数据集太少没有参考意义
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

model_c = XGBClassifier(n_estimators=20,  # 迭代次数
                       earning_rate=0.1,  # 学习率
                       max_depth=5)
model_c.fit(x_train, y_train)
result_c = model_c.predict(x_test)


# 结果比对
print(y_test)
print(result_c)