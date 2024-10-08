# _*_coding:utf-8_*_
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from plt_utils import draw_Data

# 随机生成原始数据
data_size = 50
np.random.seed(3)
x_coords = np.random.randint(0, 21, size=data_size)
y_coords = np.random.randint(0, 21, size=data_size)
# 将x坐标和y坐标组合成一个二维数组
data = np.column_stack((x_coords, y_coords))

model = KMeans(n_clusters=2, random_state=0, n_init="auto")
model.fit(data)

label = model.labels_

# 绘制分类前后的对比
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.scatter(data[:, 0], data[:, 1])
ax1.set_title("Before")
draw_Data(ax2, data, label)
ax2.set_title("After")
plt.show()
