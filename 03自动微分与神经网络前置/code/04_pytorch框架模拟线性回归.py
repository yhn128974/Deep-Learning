"""
演示
    pytorch 框架 模拟线性回归

案例工作流：
    1.准备数据集
        numpy数组 -> 张量 -> 数据集对象Dataset -> 数据加载器DataLoader（分批次加载）
    2.构建神经网络模型
        nn.Linear()线性回归模型
    3.设置损失函数和优化器
        nn.MSELoss, optim.SGD(w新 = w旧 - 学习率 * 梯度)
    4.模型训练
        1.前向传播
        2.计算损失
        3.梯度清零
        4.反向传播
        5.更新参数:w新 = w旧-学习率*梯度
    5.模型测试

"""

# 导包
import torch
# numpy对象 -> 张量 -> 数据集对象TensorDataset -> 数据加载器DataLoader
from torch.utils.data import TensorDataset  # 构造数据集对象
from torch.utils.data import DataLoader  # 数据加载器，按批次加载数据
from torch import nn  # 提供 MSE损失函数 和 线性层，线性层用来模拟线性回归模型
from torch import optim  # 提供各种优化器，用于更新模型参数，公式为：w ← w - lr * grad
from sklearn.datasets import make_regression  # 创建线性回归的示例数据集
import matplotlib.pyplot as plt

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1.准备数据集
# 定义函数，创建数据集
def create_dataset():
    # 创建数据集对象
    x, y, coef = make_regression(
        n_samples=100,          # 100条样本
        n_features=1,           # 1个特征
        n_targets=1,           # 1个标签/目标
        noise=10,           # 噪声
        bias=13.9,            # 偏移
        coef=True,          # 是否返回系数
        random_state=5          # 随机状态
    )
    # 封装成 张量
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    # 返回结果
    return x, y, coef

# 2.模型训练
def train(x,y,coef):
    # 1.准备数据集：numpy数组 -> 张量 -> 数据集对象 -> 数据加载器（分批次加载）
    # 1.1 创建数据集对象
    dataset = TensorDataset(x, y)
    # 1.2 创建数据加载器
    # 参数1：数据集对象；参数2：批次大小，一般64,32;参数3：是否打乱数据，训练时打乱，测试时不打乱；
    # 参数4：是否删除最后一个不完整的批次，一般是False
    # 100/16 = 6...4, 6+1
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=False)
    # 2.构建模型：nn.Linear()
    model = nn.Linear(1,1)
    # 3.设置损失函数和优化器
    # 损失函数
    loss_fn = nn.MSELoss()
    # 优化器,执行梯度下降法: w新 = w旧 - 学习率 * 梯度
    # 参数1: 待训练的模型参数；参数2：学习率，常用0.01,0.001
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # 4.模型训练
    # 定义训练轮次
    epochs = 100
    # 记录 每个轮次的平均损失
    total_losses = []
    # 遍历轮数epochs
    for epoch in range(epochs):
        # 1.初始化 当前轮次的训练总损失
        total_loss = 0.0
        # 2.初始化 当前轮次的总样本数
        total_samples = 0
        # 3.遍历批次，遍历数据加载器，分批训练
        for train_x, train_y in dataloader:
            # print(f"train_x:{train_x.shape}")
            # print(f"train_y:{train_y.shape}")
            # 1.前向传播，计算预测值
            y_pred = model(train_x) # (16,1)
            # print(f"y_pred: {y_pred.shape}")
            # 2.计算损失
            loss = loss_fn(y_pred, train_y.reshape(-1, 1))
            # 3.梯度清零
            optimizer.zero_grad()
            # 4.反向传播
            loss.backward()
            # 5.更新参数:w新 = w旧-学习率*梯度，更新模型的参数
            optimizer.step()
            # 6.统计 训练损失 和 样本数
            # 使用train_x.shape[0]来获取当前批次的样本数量，因为最后一个批次可能不完整
            total_loss += loss.item() * train_x.shape[0]
            total_samples += train_x.shape[0]
            ...
        # 4.计算当前轮次的平均损失
        train_loss = total_loss / total_samples
        # 5.添加当前轮次平均损失到列表中
        total_losses.append(train_loss)
        # 6.打印当前轮次的训练平均损失
        print(f"{epoch+1}/{epochs}: | "
              f"{train_loss:.4f}")
        ...
    # 5.可视化训练过程
    print(f"损失列表：{total_losses}")
    plt.figure(figsize=(15,8))
    # 绘制第一个子图：训练损失曲线图
    plt.subplot(1,2,1)
    plt.title("训练损失曲线")
    x_epoch = range(epochs)
    plt.plot(x_epoch, total_losses, label="训练损失")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid()
    plt.legend()
    # plt.show()
    # 绘制第二个子图：真实值和预测值的对比
    plt.subplot(1, 2, 2)
    plt.title("真实值和预测值的对比")
    plt.scatter(x, y, label="真实散点")
    # 计算理论的真实线性回归直线
    y_true = [v*coef.item()+13.9 for v in x]
    plt.plot(x, y_true, label="真实线性回归")
    # 计算模型预测值
    with torch.no_grad():   # 关闭梯度计算
        y_pred = model(x).detach().numpy()
    # 绘制模型预测的线性回归直线
    plt.plot(x, y_pred, label="预测值")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend()
    plt.show()

# 测试
if __name__ == '__main__':
    # 1.准备数据集
    x, y, coef = create_dataset()
    print(f"x:{x.shape}")
    print(f"y:{y.shape}")
    print(f"coef:{coef}, shape: {coef.shape}")
    # 绘制样本散点图
    plt.scatter(x, y, label="真实样本")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    # 2.模型训练
    train(x, y, coef)