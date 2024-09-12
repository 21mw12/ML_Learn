# 吴恩达机器学习

“Field of study that gives computers the ability to learn without being explicitly programmed.”——arthur Samuel (1959)

网课链接：https://www.bilibili.com/video/BV1Pa411X76s?p=1&vd_source=8566cec36593b0e28ee03f3c724b87d0

课后习题链接：https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/tree/f2757f85b99a2b800f4c2e3e9ea967d9e17dfbd8



# 分类

- 监督学习（Supervised learning）：回归、分类
- 无监督学习（Unsupervised learning）：聚类、异常检测、降维
- 强化学习（Reinforcement learning）



# 过拟合欠拟合

- 过拟合：指训练误差和测试误差之间的差距太大。模型在训练集上表现很好，但在测试集上却表现很差。模型对训练集"死记硬背"，没有理解数据背后的规律，泛化能力差。
- 欠拟合：指模型不能在训练集上获得足够低的误差。模型复杂度低，模型在训练集上就表现很差，没法学习到数据背后的规律。

- 解决方法：
  - 收集更多的数据
  - 减少使用特征
  - 正则化：减少参数的大小



# 线性回归

## 目的

通过数据计算出一条尽可能符合数据的函数


![线性回归示例](img/线性回归示例.png)



## 代价函数

使用代价函数来评判当前函数的拟合程度，代价越低拟合程度越好。（一般使用二维图像和等高线表示）

机器学习的目的就是找到代价函数的最低点，即最小值。

![代价函数](img/代价函数.png)



## 梯度下降

梯度下降算法是用来快速确定函数中的最小值。

![梯度下降](img/梯度下降.png)

当起始点再最小值的右边， 斜率为正计算出的偏导数也是正数，更新后的参数会后移靠近最小值。反之同理。

![梯度下降示例](img/梯度下降示例.png)



## 单特征值公式

函数模型：其中x和y来源于数据，w和b是待求解的参数
$$
f_{w,b}(x) = wx+b
$$
代价函数：用于评估w和b的代价，越低拟合程度越好
$$
J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}(f_{w,b} (x^{(i)})-y^{(i)})^2
$$
梯度下降算法：用于快速的计算最优的w和b（其中的alpha就是学习率）
$$
w = w - \alpha \frac{d}{dw} J(w,b) \\
b = b - \alpha \frac{d}{db} J(w,b) \\
$$
将代价函数带入梯度下降算法并计算偏导数后的表达式为：
$$
w = w - \alpha \frac{1}{m}\sum_{i=1}^{m}(f_{w,b} (x^{(i)})-y^{(i)})x^{(i)} \\
b = b - \alpha \frac{1}{m}\sum_{i=1}^{m}(f_{w,b} (x^{(i)})-y^{(i)}) \\
$$
求偏导过程如下：
$$
\frac{d}{dw} J(w,b)
& = \frac{d}{dw}\frac{1}{2m}\sum_{i=1}^{m}(f_{w,b} (x^{(i)})-y^{(i)})^2 \\
& = \frac{d}{dw}\frac{1}{2m}\sum_{i=1}^{m}(wx^{(i)}+b-y^{(i)})^2 \\
& = \frac{1}{2m}\sum_{i=1}^{m}(wx^{(i)}+b-y^{(i)})2x^{(i)} \\
& = \frac{1}{m}\sum_{i=1}^{m}(wx^{(i)}+b-y^{(i)})x^{(i)} \\
\frac{d}{db} J(w,b)
& = \frac{d}{db}\frac{1}{2m}\sum_{i=1}^{m}(f_{w,b} (x^{(i)})-y^{(i)})^2 \\
& = \frac{d}{db}\frac{1}{2m}\sum_{i=1}^{m}(wx^{(i)}+b-y^{(i)})^2 \\
& = \frac{1}{2m}\sum_{i=1}^{m}(wx^{(i)}+b-y^{(i)})2 \\
& = \frac{1}{m}\sum_{i=1}^{m}(wx^{(i)}+b-y^{(i)}) \\
$$



## 多特征值公式

函数模型：
$$
f_{\vec{w},b}(\vec{x}) = w_{1}x_{1}+...+w_{n}x_{n}+b \\
\vec{w} = [w_{1} ... w_{n}] \\
\vec{x} = [x_{1} ... x_{n}] \\
f_{\vec{w},b}(\vec{x}) = \vec{w}.\vec{x}+b
$$
代价函数：
$$
J(\vec{w},b)=\frac{1}{2m}\sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x})-y^{(i)})^2
$$
梯度下降算法：
$$
w_{j} = w_{j} - \alpha \frac{d}{dw_{j}} J(\vec{w},b) \\
b = b - \alpha \frac{d}{db} J(\vec{w},b) \\
$$
将代价函数带入梯度下降算法并计算偏导数后的表达式为：
$$
w_{n} = w_{n} - \alpha \frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})x_{n}^{(i)} \\
b = b - \alpha \frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)}) \\
$$



## 特征值缩放

特征值的大小会显著的影响到各自w系数的大小，为了提高机器学习的性能，避免特征值过大或过小，需要对特征值进行缩放。

![特征值缩放](img/特征值缩放.png)

名词解释：可以认为特征缩放包括归一化和标准化。



### 特征缩放

Feature Scaling
$$
x_{i, scaled} = \frac{x_{i}}{x_{max}}
$$


### 归一化

Normalization
$$
特征值的均值:\mu_{i} \\
x_{i, scaled} = \frac{x_{i} - \mu_{i}}{x_{max} - x_{min}}
$$


### 标准化

Standardization (Z-Score Normalization)
$$
特征值的标准差:\sigma_{i} \\
特征值的平均值:\bar{x}_{i} \\
x_{i, scaled} = \frac{x_{i} - \bar{x}_{i}}{\sigma_{i}}
$$


## 设置学习率

对于学习率（0~1之间），若太小每次更新的距离太短导致迭代时间太长，若大太则有可能直接跨过最低点到另一边从而无法收敛。

通过设置不同的学习率，并绘制部分迭代过程中的代价值图像，通过观察图像再选择一条最大的可收敛的曲线作为最后的学习率

>如：0.001   0.003   0.01   0.03   0.1   0.3   1



## 多项式回归

如果要拟合曲线则需要使用多项式的函数模型，如：
$$
f_{\vec{w},b}(x) = w_{1}x+w_{2}x^{2}+w_{3}x^{3}+...+b \\
f_{\vec{w},b}(x) = w_{1}x+w_{2}\sqrt{x}+b \\
$$



## 正则化

代价函数：
$$
J(\vec{w},b)=\frac{1}{2m}\sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x})-y^{(i)})^2+\frac{\lambda}{2m}\sum_{j=1}^{n}w_{j}^{2}
$$
梯度下降算法：
$$
w_{j} = w_{j} - \alpha 
\left \{ 
\frac{1}{m}\sum_{i=1}^{m}[(f_{\vec{w},b} (\vec{x}^{(i)})-y^{(i)})x_{j}^{(i)}]+\frac{\lambda}{m}w_{j}
\right \}  \\

b = b - \alpha \frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)}) \\
$$
w项化简：
$$
w_{j} = w_{j}(1-\alpha\frac{\lambda}{m}) - \alpha \frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w},b} (\vec{x}^{(i)})-y^{(i)})x_{j}^{(i)}
$$


# 逻辑回归

## 目的





## 逻辑函数

不同与线性回归输出的值没有固定的范围，逻辑回归的输出有固定的几个值。如果使用函数输出固定值就需要用逻辑函数。
$$
g(z) = \frac{1}{1 + e^{-z}} \\
0 < g(z) < 1
$$
![逻辑函数](img/逻辑函数.png)



## 逻辑回归模型

$$
z = \vec{w}.\vec{x} + b \\
f_{\vec{w},b}(\vec{x}) = g(\vec{w}.\vec{x} + b) = \frac{1}{1 + e^{-(\vec{w}.\vec{x} + b)}}
$$

通常会设置一个值(threshold)，当逻辑函数的结果大于这个值则输出1，反之则输出0
$$
is \ f_{\vec{w},b}(\vec{x}) \ \ge \ threshold? \\
Yes: y = 1 \\
No: y = 0
$$

$$
When \ is \ f_{\vec{w},b}(\vec{x}) \ \ge \ threshold? \\
g(z) \ge threshold \\
z  \ge threshold \\
\vec{w}.\vec{x} + b \ge threshold \\
y = 1 \\
\vec{w}.\vec{x} + b < threshold \\
y = 0 \\
$$

其中z的表达式就是边界函数，需要根据不同的情况进行调整。



## 逻辑回归公式

函数模型：
$$
f_{\vec{w},b}(\vec{x}) = g(\vec{w}.\vec{x} + b) = \frac{1}{1 + e^{-(\vec{w}.\vec{x} + b)}}
$$
代价函数：
$$
J(\vec{w},b)=\frac{1}{m}\sum_{i=1}^{m}L(f_{\vec{w},b}(\vec{x}^{(i)},y^{(i)}) \\
L(f_{\vec{w},b}(\vec{x}^{(i)},y^{(i)}) = \left\{
\begin{aligned}
x = -log(f_{\vec{w},b}(\vec{x}^{(i)}) \quad\quad if \ y^{(i)}=1 \\
y = -log(1-f_{\vec{w},b}(\vec{x}^{(i)}) \quad if \ y^{(i)}=0 \\
\end{aligned}
\right.
$$

$$
J(\vec{w},b)=-\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}log(f_{\vec{w},b}(\vec{x}^{(i)})+(1-y^{(i)})log(1-f_{\vec{w},b}(\vec{x}^{(i)})] \\
$$

梯度下降算法：
$$
w_{j} = w_{j} - \alpha \frac{d}{dw_{j}} J(\vec{w},b) \\
b = b - \alpha \frac{d}{db} J(\vec{w},b) \\
$$
将代价函数带入梯度下降算法并计算偏导数后的表达式为：
$$
w_{n} = w_{n} - \alpha \frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})x_{n}^{(i)} \\
b = b - \alpha \frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)}) \\
$$



## 正则化

代价函数：
$$
J(\vec{w},b)=-\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}log(f_{\vec{w},b}(\vec{x}^{(i)})+(1-y^{(i)})log(1-f_{\vec{w},b}(\vec{x}^{(i)})] + \frac{\lambda}{2m}\sum_{j=1}^{n}w_{j}^{2} \\
$$
梯度下降算法：
$$
w_{j} = w_{j} - \alpha 
\left \{ 
\frac{1}{m}\sum_{i=1}^{m}[(f_{\vec{w},b} (\vec{x}^{(i)})-y^{(i)})x_{j}^{(i)}]+\frac{\lambda}{m}w_{j}
\right \}  \\

b = b - \alpha \frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)}) \\
$$

















