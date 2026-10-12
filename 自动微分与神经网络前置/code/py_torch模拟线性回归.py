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

# 定义函数，创建线性回归样本数据
"""
make_regression 中设置 coef=True 后，函数会返回生成数据时使用的真实系数（权重）。

对于你的例子（n_features=1），返回的 coef 就是一个标量，表示 y = coef * x + noise 中的那个 真实斜率。
"""


#  numpy对象=>tenser张量对象=>数据集对象TensorDataset =>数据加载器DataLoader（分批次加载）
def create_dateset():
    x,y,coef=make_regression(
        n_samples=100,
        n_features=1,
        noise=10, 
        coef=True,
        bias=5,
        random_state=42
    )
    # 先吧numpy对象转换为tenser张量对象
    x=torch.tensor(x,dtype=torch.float32)
    y=torch.tensor(y,dtype=torch.float32)

    return x,y,coef

def train_data(x,y,coef):
    # tenser张量对象转化为数据集对象TensorDataset
    dataset=TensorDataset(x,y)

    # 数据集对象TensorDataset转化为数据加载器DataLoader（分批次加载）
    # DataLoader参数说明：
        # dataset:数据集对象
        # batch_size:批次大小
        # shuffle:是否打乱数据（训练集打乱，测试集不打乱）
        # drop_last:是否丢弃最后一个不完整的批次
    dataloader=DataLoader(dataset,batch_size=10,shuffle=True)

    # 创建线性回归模型
    model=nn.Linear(1,1)

    # 创建损失函数对象
    criterion=nn.MSELoss()

    # 创建优化器对象
    optimizer=optim.SGD(model.parameters(),lr=0.01)

    # 开始训练
        # 训练轮数
        # 每轮的平均损失值
        # 训练总损失
        # 训练样本数
    epochs,loss_list,total_loss,train_sets_count=100,[],0,0

    for i in range(epochs): #训练100轮

        for tr_x,tr_y in dataloader: #100个数据分成10批，循环会执行 10 次，也就是每批10个数据进行训练
            y_pred=model(tr_x) #1.前向传播（正向传播），计算出 预测值（z）

            #2.计算该批数据的平均函数（MSE）损失值 reshape(-1,1):将数据变为1列
            loss=criterion(y_pred,tr_y.reshape(-1,1)) 
            
            #记录训练损失
            total_loss+=loss.item()  

            # 记录训练批次数
            train_sets_count+=1

            # 模型参数清零：在开始一轮新的反向传播之前，必须将上一轮计算得到的梯度清零，否则梯度会累积
            optimizer.zero_grad()

            # 反向传播计算梯度
            loss.backward()
            
            # 5.更新参数:w新 = w旧-学习率*梯度，更新模型的参数
            optimizer.step()
        
        # 计算当前轮次的平均损失值并记录
        loss_list.append(total_loss/train_sets_count)
        # 打印每轮训练后的平均损失值
        print(f"第{i}轮训练后的平均损失值为：{loss_list[i]}")    

    # 打印最终结果
    print(f"真实斜率为{coef},真实偏执为:5")
    print(f"模型学到的斜率为{model.weight.item():.5f},模型的偏执为:{model.bias.item():.5f}")

    
    # 绘制训练过程中的损失曲线图
    plt.plot(loss_list)
    plt.xlabel("epochs") # 横坐标：表示训练的轮数（epochs）
    plt.ylabel("loss") # 纵坐标：表示损失值
    plt.title("loss curve") # 图表的标题
    plt.show()


if __name__=='__main__':
    # 调用函数，进行数据创建
    x,y,coef=create_dateset()
    print(f"真实系数（斜率）: {coef} x:{x}, y:{y}")
    # 模型训练
    train_data(x,y,coef)

            
            

        


        
    
    
    













if __name__ == "__main__":
    # 定义数据
    x,y,coef=create_dateset()
    print(f"真实系数（斜率）: {coef} x:{x}, y:{y}")
    # 模型训练
    train_data(x,y,coef)

 
