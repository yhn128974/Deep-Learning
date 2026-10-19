"""
演示
    dropout随机失活

注意：
    dropout是一个特殊的网络层，没有可训练的参数，只有一个超参数p。dropout层通常添加到输出层之前

正则化作用：
    缓解模型过拟合，提高模型泛化能力
正则化的方法：
    L1正则化：在损失函数中添加绝对值项|w|, 权重可以为0，相当于降维
    L2正则化：在损失函数中添加平方项w^2, 权重可以接近0，Adam/AdamW权重衰减，对离散值敏感

    Dropout随机失活：
        1.训练时，model.train(),随机以概率p让一部分神经元失活，输出设置为0。没有失活的神经元的输出*1/(1-p)
        2.测试时，model.eval(),关闭dropout随机失活，让所有神经元一起参与

需要掌握：
    nn.Dropout(p=0.1)
    参数p设置：0<p<1，通常p<0.5, 当模型参数较多 数据量较大 时，可以适当增加p,比如0.4~0.5
"""
# 导包
import torch
import torch.nn as nn

# 1.定义函数，演示 dropout随机失活
def demo01():
    # 0.设置随机种子
    torch.manual_seed(50)
    # 1.创建输入数据，(batch_size,input_dim)
    x = torch.randint(0,10,(1,5),dtype=torch.float32)
    print(f"x: {x},x.shape: {x.shape}")

    # 创建神经网络，一个简单的线性层
    linear1 = nn.Linear(5,10)
    x = linear1(x)  # (1,5) -> (1,10)
    print(f"x: {x},x.shape: {x.shape}")

    # 进入relu激活函数
    x=torch.relu(x)
    print(f"x: {x},x.shape: {x.shape}")


    # 3.创建一个dropout层，来实现dropout随机失活，并对激活值y进行y(1-p)缩放减小平均值的变化
    dropout = nn.Dropout(p=0.5)
    x = dropout(x)
    print(f"x: {x},x.shape: {x.shape}")
    ...

# 测试
if __name__ == '__main__':
    demo01()
