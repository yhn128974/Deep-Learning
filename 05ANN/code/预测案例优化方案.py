"""
模型调优思路：
1. 对输入数据进行标准化
2. 调整优化器
3. 调整学习率
4. 增加批量归一化层BN1D
5. 增加网络层数、神经元个数
6. 增加dropout层，缓解过拟合
7. 增加训练轮数
8. 使用GPU加速训练
9. ...

"""

import os
import torch  # pytorch框架
from torch.utils.data import TensorDataset  # 数据集对象
from torch.utils.data import DataLoader  # 数据加载器
import torch.nn as nn  # 神经网络层的操作
import torch.optim as optim  # 优化器
from sklearn.model_selection import train_test_split  # 拆分 训练集 和 测试集
import matplotlib.pyplot as plt  # 绘图
import numpy as np  # 数组操作
import pandas as pd  # 数据处理
import time  # 时间模块

from torchsummary import summary
from sklearn.preprocessing import StandardScaler
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False            # 解决负号显示问题
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备：{device}")

# 1.准备数据集
def create_dataset():
    # 1.加载CSV数据集（使用 __file__ 构建绝对路径，不受运行目录影响）
    _base = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(_base, '..', 'data', '手机价格预测.csv'))
    # 2.获取特征x 和 标签y
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # 3.特征列 转换为 float
    x = x.astype(np.float32)

    # 4.拆分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=6, stratify=y)

    # 数据标准化
    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.transform(x_test)
    # 5.把数据集转换为张量, numpy数组 -> 张量 -> 数据集对象TensorDataset -> 数据加载器DataLoader
    train_dataset = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))

    # 6.返回结果
    return train_dataset, test_dataset, x_train.shape[1], len(y_train.unique())

# 2.搭建神经网络
class PhonePriceClassifier(nn.Module):
    # 1.定义__init__方法, 初始化父类成员，搭建神经网络
    def __init__(self, input_dim, output_dim):
        # 初始化父类成员
        super().__init__()
        # 搭建神经网络
        # 优化1：增加网络深度+使用BN1D层
        self.linear1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.linear4 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(0.1)

    # 2.定义forward方法，实现前向传播
    def forward(self, x):
        # 隐藏层1, linear -> BN层 -> 激活函数 -> dropout
        x = torch.relu(self.bn1(self.linear1(x)))
        x = self.dropout(x)
        # 隐藏层2
        x = torch.relu(self.bn2(self.linear2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.linear3(x)))
        x = self.dropout(x)
        # 输出层
        x = self.linear4(x)
        # 返回结果
        return x

# 3.模型训练
def train(train_dataset, input_dim, output_dim):
    # 1.创建数据加载器DataLoader
    # 参数1：数据集对象；参数2：批量大小；参数3：是否打乱数据
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # 2.创建神经网络模型
    model = PhonePriceClassifier(input_dim, output_dim)
    model.to(device)
    # 3.定义损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 4.创建优化器对象
    # 优化3：使用Adam优化方法
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    # 优化4：添加学习率调度，optim.lr_scheduler.CosineAnnealingWarmRestarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5*len(train_loader), T_mult=1)
    # 5.模型训练
    # 定义变量，记录 训练总轮数
    epochs = 50
    # 定义遍历，记录每轮次的损失和准确率
    train_losses = []
    train_accs = []
    # 遍历轮数
    for epoch in range(epochs):
        # 记录 每次训练的 损失值，训练样本数，预测正确的样本数
        total_loss = 0.
        total_sample = 0
        total_correct = 0
        # 记录训练开始时间
        start_time = time.time()
        # 遍历批次
        for x, y in train_loader:
            # 训练模式
            model.train()
            # 移动数据到GPU
            x = x.to(device)
            y = y.to(device)
            # 前向传播
            y_pred = model(x)
            # 计算损失值
            loss = loss_fn(y_pred, y)
            # 更新总损失
            batch_size = x.shape[0]
            total_loss += loss.item() * batch_size
            # 更新总样本数
            total_sample += batch_size
            # 更新预测正确的样本数
            total_correct += (torch.argmax(y_pred, dim=-1) == y).sum().item()
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            scheduler.step()

        # 打印训练信息
        print(
            f"epoch: {epoch}, loss: {total_loss / total_sample}, acc: {total_correct / total_sample}, time: {time.time() - start_time:.2f}")
        train_losses.append(total_loss / total_sample)
        train_accs.append(total_correct / total_sample)
    # 6.保存模型参数
    os.makedirs("./model", exist_ok=True)
    # print(f"\n\n模型的参数信息: {model.state_dict()}\n\n")
    torch.save(model.state_dict(), "./model/phone_price_classifier.pth")  # 模型文件，后缀名 .pth, .pkl, .pickle 均可
    # 7.返回训练结果
    return train_losses, train_accs

# 可视化训练过程
def plot_train_process(train_losses, train_accs):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title("训练损失")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(train_accs)
    plt.title("训练准确率")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 4.模型测试
@torch.no_grad()
def evaluate(test_dataset, input_dim, output_dim):
    # 1. 创建神经网络分类对象.
    model = PhonePriceClassifier(input_dim, output_dim)
    model.to(device)
    # 2.加载模型参数
    model.load_state_dict(torch.load('./model/phone_price_classifier.pth'))
    # 3.创建数据加载器DataLoader
    # 参数1：数据集对象；参数2：批量大小；参数3：是否打乱数据
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # 4.记录正确样本的数量
    correct_sample = 0
    total_sample = 0
    # 5.开始测试, 遍历数据加载器的批次数据
    for x, y in test_loader:
        # 切换到测试模式
        model.eval()
        # 张量 迁移 到 GPU
        x = x.to(device)
        y = y.to(device)
        # 6.前向传播，模型预测
        y_pred = model(x)  # 形状(batch_size, 4), 每个类别的输出logits
        # print(f"预测结果y_pred：{y_pred}")
        # 7.计算预测正确的数量
        correct = y_pred.argmax(dim=1) == y
        correct_sample += correct.sum().item()
        total_sample += y.size(0)
    print(f"测试集准确率：{correct_sample / total_sample}")

# 5.测试
if __name__ == '__main__':
    # 1.准备数据集
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    # 打印
    print(f"train_dataset: {train_dataset}")
    print(f"test_dataset: {test_dataset}")
    print(f"input_dim: {input_dim}")
    print(f"output_dim: {output_dim}")
    # 2.构建神经网络
    model = PhonePriceClassifier(input_dim, output_dim)
    # 3.统计模型参数
    model.to(device)
    # summary(model,input_size=(input_dim,))
    # 4.模型训练
    train_losses, train_accs = train(train_dataset, input_dim, output_dim)
    plot_train_process(train_losses, train_accs)
    # 5.模型测试
    evaluate(test_dataset, input_dim, output_dim)