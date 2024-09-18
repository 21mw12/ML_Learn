# _*_coding:utf-8_*_
import math

# 创建数据
def createDataSet():
    # 数据
    dataSet = [
        [0, 0, 0, 0, 'no'],
        [0, 0, 0, 1, 'no'],
        [0, 1, 0, 1, 'yes'],
        [0, 1, 1, 0, 'yes'],
        [0, 0, 0, 0, 'no'],
        [1, 0, 0, 0, 'no'],
        [1, 0, 0, 1, 'no'],
        [1, 1, 1, 1, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [2, 0, 1, 2, 'yes'],
        [2, 0, 1, 1, 'yes'],
        [2, 1, 0, 1, 'yes'],
        [2, 1, 0, 2, 'yes'],
        [2, 0, 0, 0, 'no'],
    ]
    # 列名
    labels = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']
    return dataSet, labels

# 获取当前样本里最多的标签
def getMaxLabelByDataSet(curLabelList):
    classCount = {}
    maxKey, maxValue = None, None
    for label in curLabelList:
        if label in classCount.keys():
            classCount[label] += 1
            if maxValuex < classCount[label]:
                maxKey, maxValue = label, classCount[label]
        else:
            classCount[label] = 1
            if maxKey is None:
                maxKey, maxValue = label, 1
    return maxKey

# 计算熵值
def calcEntropy(dataSet):
    # 1. 获取所有样本数
    exampleNum = len(dataSet)
    # 2. 计算每个标签值的出现数量
    labelCount = {}
    for featVec in dataSet:
        curLabel = featVec[-1]
        if curLabel in labelCount.keys():
            labelCount[curLabel] += 1
        else:
            labelCount[curLabel] = 1
    # 3. 计算熵值(对每个类别求熵值求和)
    entropy = 0
    for key, value in labelCount.items():
        # 概率值
        p = labelCount[key] / exampleNum
        # 当前标签的熵值计算并追加
        curEntropy = -p * math.log(p, 2)
        entropy += curEntropy
    # 4. 返回
    return entropy

# 选择最好的特征进行分割，返回最好特征索引
def chooseBestFeatureToSplit(dataSet):
    # 1. 计算特征个数 -1 是减去最后一列标签列
    featureNum = len(dataSet[0]) - 1
    # 2. 计算当前（未特征划分时）熵值
    curEntropy = calcEntropy(dataSet)
    # 3. 找最好特征划分
    bestInfoGain = 0  # 最大信息增益
    bestFeatureIndex = -1  # 最好特征索引
    for i in range(featureNum):
        # 拿到当前列特征
        featList = [example[i] for example in dataSet]
        # 获取唯一值
        uniqueVals = set(featList)
        # 新熵值
        newEntropy = 0
        # 计算分支（不同特征划分）的熵值
        for val in uniqueVals:
            # 根据当前特征划分dataSet
            subDataSet = splitDataSet(dataSet, i, val)
            # 加权概率值
            weight = len(subDataSet) / len(dataSet)
            # 计算熵值，追加到新熵值中
            newEntropy += (calcEntropy(subDataSet) * weight)
        # 计算信息增益
        infoGain = curEntropy - newEntropy
        # 更新最大信息增益
        if bestInfoGain < infoGain:
            bestInfoGain = infoGain
            bestFeatureIndex = i
    # 4. 返回
    return bestFeatureIndex

# 根据当前选中的特征和唯一值去划分数据集
def splitDataSet(dataSet, featureIndex, value):
    returnDataSet = []
    for featVec in dataSet:
        if featVec[featureIndex] == value:
            # 将featureIndex那一列删除
            deleteFeatVec = featVec[:featureIndex]
            deleteFeatVec.extend(featVec[featureIndex + 1:])
            # 将删除后的样本追加到新的dataset中
            returnDataSet.append(deleteFeatVec)
    return returnDataSet


# 递归生成决策树节点
def createTreeNode(dataSet, labels, featLabels):
    # 取出当前节点的样本的标签 -1 表示在最后一位
    curLabelList = [example[-1] for example in dataSet]

    # -------------------- 停止条件 --------------------
    # 1. 判断当前节点的样本的标签是不是已经全为1个值了，如果是则直接返回其唯一类别
    if len(curLabelList) == curLabelList.count(curLabelList[0]):
        return curLabelList[0]
    # 2. 判断当前可划分的特征数是否为1，如果为1则直接返回当前样本里最多的标签
    if len(labels) == 1:
        return getMaxLabelByDataSet(curLabelList)

    # -------------------- 下面是正常选择特征划分的步骤 --------------------
    # 1. 选择最好的特征进行划分(返回值为索引)
    bestFeatIndex = chooseBestFeatureToSplit(dataSet)
    # 2. 利用索引获取真实值
    bestFeatLabel = labels[bestFeatIndex]
    # 3. 将特征划分加入当前决策树
    featLabels.append(bestFeatLabel)
    # 4. 构造当前节点
    myTree = {bestFeatLabel: {}}
    # 5. 删除被选择的特征
    del labels[bestFeatIndex]
    # 6. 获取当前最佳特征的那一列
    featValues = [example[bestFeatIndex] for example in dataSet]
    # 7. 去重(获取唯一值)
    uniqueFeaValues = set(featValues)
    # 8. 对每个唯一值进行分支
    for value in uniqueFeaValues:
        # 递归创建树
        myTree[bestFeatLabel][value] = createTreeNode(
            splitDataSet(dataSet, bestFeatIndex, value), labels.copy(),
            featLabels.copy())
    # 9. 返回
    return myTree

# 测试一下！！！
# 1. 获取数据集
dataSet,labels = createDataSet()
# 2. 构建决策树
myDecisionTree = createTreeNode(dataSet,labels,[])
# 3. 输出
print(myDecisionTree)