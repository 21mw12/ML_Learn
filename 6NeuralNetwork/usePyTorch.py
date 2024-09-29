# _*_coding:utf-8_*_
"""
 使用pytorch实现的手写数字识别
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class MyMnistData(Dataset):
    def __init__(self, filePath):
        self.filePath = filePath
        self.datas = np.array(pd.read_csv(filePath, header=None))
        self.items = self.datas[:, 1:].astype(np.float32).reshape(-1, 28, 28)  # 将数据转换为28x28的图像
        self.labels = self.datas[:, :1].astype(np.int64).reshape(-1)  # 将标签转换为整数

    def __getitem__(self, idx):
        # 重写获取单个数据的函数
        image = self.items[idx]
        label = self.labels[idx]
        return image, label

    def __len__(self):
        # 重写获取数据总数的函数
        return len(self.datas)

class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        # 三层线性连接
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 将图像展平为一维向量
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def test(model, test_loader):
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, labels = data
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'最终得分: {correct / total}%')

if __name__ == '__main__':
    # 参数设置
    learning_rate = 0.003   # 学习率
    epochs = 5              # 世代次数，即训练次数

    # 加载训练数据
    train_data = MyMnistData("data/mnist_train.csv")
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
    # 加载测试数据
    test_data = MyMnistData("data/mnist_test.csv")
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

    # 创建模型
    model = LinearNet()
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 使用Adam优化器

    # 训练模型
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'世代 [{epoch + 1}/{10}], 步骤 [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # 测试模型
    test(model, test_loader)