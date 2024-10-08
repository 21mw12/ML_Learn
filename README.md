“Field of study that gives computers the ability to learn without being explicitly programmed.”——arthur Samuel (1959)



# 介绍

在学习机器学习过程中编写的代码以及编写的笔记。



# 学习途径

- 学习路线：[纯新手自学入门机器/深度学习指南（附一个月速成方案） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/37349519)
- 网课链接
  - [吴恩达机器学习](https://www.bilibili.com/video/BV1Pa411X76s?p=1&vd_source=8566cec36593b0e28ee03f3c724b87d0)
  - [周志华老师西瓜书](https://www.bilibili.com/video/BV1gG411f7zX?p=1)
  - [3Blue1Brown深度学习]([【官方双语】深度学习之神经网络的结构 Part 1 ver 2.0_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1bx411M7Zx/?spm_id_from=333.999.0.0&vd_source=8566cec36593b0e28ee03f3c724b87d0))
- 资料来源
  - [吴恩达机器学习实验室]([kaieye/2022-Machine-Learning-Specialization (github.com)](https://github.com/kaieye/2022-Machine-Learning-Specialization/tree/main))
  - [吴恩达机器学习笔记]([fengdu78/Coursera-ML-AndrewNg-Notes: 吴恩达老师的机器学习课程个人笔记 (github.com)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/tree/master))
  - [机器学习中的数学——激活函数](https://blog.csdn.net/hy592070616/article/details/120617176)
  - 神经网络和深度学习的在线书籍
    - 英文版：[Neural networks and deep learning](http://neuralnetworksanddeeplearning.com/)
    - 中文版：[引言 | 神经网络与深度学习 (gitbooks.io)](https://tigerneil.gitbooks.io/neural-networks-and-deep-learning-zh/content/)
  - [Christopher Olah的博客](https://colah.github.io/)



# 项目结构

- Note.pdf：学习笔记
- 1LinearRegression：线性回归
  - data：数据集
  - singleFeature.py：单特征值的线性回归
  - multipleFeature：多特征值的线性回归
  - ex1.py：练习1
  - sk.py：使用sklearn实现的线性回归
  - equation.py：线性回归相关的公式代码
  - lab_utils_uni.py：源自吴恩达机器学习实验室可视化相关功能
- 2LogisticRegression：逻辑回归
  - data：数据集
  - LogisticRegression.py：逻辑回归
  - ex1.py：练习1
  - ex1_result：因练习1过长，直接展示其训练结果
  - sk.py：使用sklearn实现的逻辑回归
  - equation.py：逻辑回归相关的公式代码
  - plt_utils.py：可视化相关功能
- 3DecisionTree：决策树
  - DecisionTree.py：使用sklearn实现的决策树分类和回归
  - RandomForest.py：使用sklearn实现的随机森林分类和回归
  - XGBoost.py：使用XGBoost的分类和回归
- 4BayesClassifier：贝叶斯分类器
- 5SVM：支持向量机
- 6NeuralNetwork：神经网络
  - data：数据集，下载连接如下
    - [完整训练集 mnist_train.csv](http://www.pjreddie.com/media/files/mnist_train.csv)
    - [完整测试集 mnist_train.csv](http://www.pjreddie.com/media/files/mnist_test.csv)
    - [部分训练集 mnist_train_100.csv](https://raw.githubusercontent.com/makeyourownneuralnetwork/makeyourownneuralnetwork/master/mnist_dataset/mnist_train_100.csv)
    - [部分测试集 mnist_train_10.csv](https://raw.githubusercontent.com/makeyourownneuralnetwork/makeyourownneuralnetwork/master/mnist_dataset/mnist_test_10.csv)
  - usePython.py：使用纯Python编写的手写数字识别
  - usePyTorch.py：使用PyTorchb编写的手写数字识别
- 7KMeans：K-means聚类算法
  - KMeans.py：使用sklearn实现的K-means聚类算法
  - plt_utils.py：可视化相关功能