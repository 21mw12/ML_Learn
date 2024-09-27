# _*_coding:utf-8_*_
"""
 根据《Python神经网络编程》所写
"""
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置Numpy不显示科学计数并只保留4位小数
np.set_printoptions(suppress=True, precision=4)

class neuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learnIngrate):
        """
        神经网络初始化
        :param inputNodes: 输入层结点数量
        :param hiddenNodes: 隐藏层节点数量
        :param outputNodes: 输出层节点数量
        :param learnIngrate: 学习率
        """
        # 设置结点数量
        self.iNodes = inputNodes
        self.hNodes = hiddenNodes
        self.oNodes = outputNodes

        # 设置学习率
        self.lr = learnIngrate

        '''
        随机生成权值
        使用正态分布的形式生成初始权值
         - 中心点设为0.0
         - 以下一层的节点数的开方作为标准方差来初始化全职
         - 设置矩阵大小
        ItoHW：input层到hidden层的权值
        HtoOW：hidden层到output层的权值
        '''
        self.ItoHW = np.random.normal(0.0, pow(self.hNodes, -0.5), (self.hNodes, self.iNodes))
        self.HtoOW = np.random.normal(0.0, pow(self.oNodes, -0.5), (self.oNodes, self.hNodes))

        # 设置激活函数
        self.activationFunction = lambda x: scipy.special.expit(x)

    def train(self, inputList, targetList):
        """
        反向传播训练
        :param inputList: 输入层的数据
        :param targetList: 目标结果
        """
        # 将输入转为矩阵
        inputs = np.array(inputList, ndmin=2).T
        targets = np.array(targetList, ndmin=2).T

        # 隐藏层操作
        hiddenInput = np.dot(self.ItoHW, inputs)
        hiddenOutput = self.activationFunction(hiddenInput)

        # 输出层操作
        finalInput = np.dot(self.HtoOW, hiddenOutput)
        finalOutput = self.activationFunction(finalInput)

        # 计算每层误差
        outputError = targets - finalOutput
        hiddenError = np.dot(self.HtoOW.T, outputError)

        # 更新隐藏层到输出层的权值
        self.HtoOW += self.lr * np.dot(outputError * finalOutput * (1.0 - finalOutput), np.transpose(hiddenOutput))
        # 更新输入层到隐藏层的权值
        self.ItoHW += self.lr * np.dot(hiddenError * hiddenOutput * (1.0 - hiddenOutput), np.transpose(inputs))

    def query(self, inputList):
        """
        向前传播
        :param inputList:
        :return:
        """
        # 将输入转为矩阵
        inputs = np.array(inputList, ndmin=2).T

        # 隐藏层操作
        hiddenInput = np.dot(self.ItoHW, inputs)
        hiddenOutput = self.activationFunction(hiddenInput)

        # 输出层操作
        finalInput = np.dot(self.HtoOW, hiddenOutput)
        finalOutput = self.activationFunction(finalInput)

        return finalOutput

def drawNumberPicture(data):
    """
    绘制MNIST数据中的一个数字数据
    :param data: 代表一个数字的数据
    """
    image = np.asfarray(data.tolist()[1:]).reshape(28, 28)
    plt.imshow(image, cmap="Greys", interpolation="None")
    plt.show()

if __name__ == '__main__':
    # 参数设置
    input_nodes = 784       # 输入层节点数量
    hidden_node = 100       # 隐藏层节点数量
    output_node = 10        # 输出节点数量
    learning_rate = 0.3     # 学习率
    epochs = 5              # 世代次数，即训练次数

    # 读取数据
    # trainData = np.array(pd.read_csv("data/mnist_train_100.csv", header=None))
    # testData = np.array(pd.read_csv("data/mnist_test_10.csv", header=None))
    trainData = np.array(pd.read_csv("data/mnist_train.csv", header=None))
    testData = np.array(pd.read_csv("data/mnist_test.csv", header=None))
    # drawNumberPicture(test10Data[0])

    # 构建神经网络
    NN = neuralNetwork(input_nodes, hidden_node, output_node, learning_rate)
    # print(f"输入层有{NN.iNodes}个节点-输入层有{NN.hNodes}个节点-输入层有{NN.oNodes}个节点-学习率为{NN.lr}")
    # print(NN.query([1.0, 0.5, -1.5]))

    for e in range(epochs):
        # 遍历所有训练集中的数
        for example in trainData:
            # 因为逻辑函数的原因，将所有0~255的数值压缩到0.01~1
            inputs = (np.asfarray(example[1:]) / 255 * 0.99) + 0.01
            # 设定目标结果
            targets = np.zeros(output_node) + 0.01
            targets[int(example[0])] = 0.99
            # 进行训练
            NN.train(inputs, targets)

    scoreCard = []
    for example in testData:
        target = int(example[0])
        inputs = (np.asfarray(example[1:]) / 255 * 0.99) + 0.01
        outputs = NN.query(inputs)
        label = np.argmax(outputs)
        print(f"正确答案位为{target} - 神经网认为{label}")
        if target == label:
            scoreCard.append(1)
        else:
            scoreCard.append(0)
    score = np.array(scoreCard)
    print(f"最终得分：{score.sum() / score.size}")



