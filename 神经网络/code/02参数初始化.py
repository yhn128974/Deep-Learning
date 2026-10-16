"""
参数初始化的目的：
    1.防止梯度消失、梯度爆炸
    2.加速模型收敛
    3.打破对称性，否则所有神经元更新的参数都一样

梯度消失
    梯度传播到浅层时已经变得很小，导致浅层参数更新缓慢，网络退化

梯度爆炸
    梯度传播到浅层时变得很大，导致网络参数更新过大，网络不稳定

常见参数初始化方法
    无法打破对称性的方式：
        全零，全一，固定值
    可以打破对称性的方式：
        随机数初始化：高斯分布，均匀分布
        Xavier 初始化：
        He-kaiming 初始化：

重点：
    1.kaiming初始化，Xavier初始化，全零初始化
    2.选择合适的初始化方法
        对应激活函数RELU及系列(leakyrelu,pRelu,GELU) -> kaiming
        对应激活函数非RELU及系列(sigmoid,tanh) -> xavier
        如果是浅层网络：可以考虑随机数初始化(高斯分布，均匀分布)

"""

import torch.nn as nn
# 1.定义函数，演示 均匀分布初始化
def demo01():
    # 1.创建线性层 :输入维度5，输出维度3
    model=nn.Linear(5,3)
    # 2.对权重和偏置进行均匀分布初始化
    nn.init.uniform_(model.weight,0,1)
    nn.init.uniform_(model.bias)
    # 3.打印权重和偏置
    print(f"权重{model.weight}")
    print(f"偏置{model.bias}")

# 2.定义函数，演示 正态分布初始化
def demo02():
    pass

# 3.定义函数，演示 全零 初始化
def demo03():
    # 1.创建线性层 :输入维度5，输出维度3
    model=nn.Linear(5,3)
    # 2.对权重和偏置进行均匀分布初始化
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    # 3.打印权重和偏置
    print(f"权重{model.weight}")
    print(f"偏置{model.bias}")

# 4.定义函数，演示 全一 初始化
def demo04():
    # 1.创建线性层 :输入维度5，输出维度3
    model=nn.Linear(5,3)
    # 2.对权重和偏置进行均匀分布初始化
    nn.init.ones_(model.weight)
    nn.init.ones_(model.bias)
    # 3.打印权重和偏置
    print(f"权重{model.weight}")
    print(f"偏置{model.bias}")

# 5.定义函数，演示 固定值 初始化
def demo05():
    # 1.创建线性层 :输入维度5，输出维度3
    model=nn.Linear(5,3)
    # 2.对权重和偏置进行均匀分布初始化
    nn.init.constant_(model.weight,3)
    nn.init.constant_(model.bias,3)
    # 3.打印权重和偏置
    print(f"权重{model.weight}")
    print(f"偏置{model.bias}")

# 6.定义函数，演示 均匀/正态 kaiming初始化
def demo06():
    # 1.创建线性层 :输入维度5，输出维度3
    model=nn.Linear(5,3)
    # 2.对权重和偏置进行均匀分布初始化
    # nn.init.kaiming_uniform_(model.weight)
    nn.init.kaiming_normal_(model.weight)
    nn.init.constant_(model.bias,0)
    # 3.打印权重和偏置
    print(f"权重{model.weight}")
    print(f"偏置{model.bias}")



# 7.定义函数，演示 xavier初始化
def demo07():
    # 1.创建线性层 :输入维度5，输出维度3
    model=nn.Linear(5,3)
    # 2.对权重和偏置进行均匀分布初始化
    nn.init.xavier_uniform_(model.weight)
    # nn.init.xavier_normal_(model.weight)
    nn.init.constant_(model.bias,0)
    # 3.打印权重和偏置
    print(f"权重{model.weight}")
    print(f"偏置{model.bias}")
    

# 测试
if __name__ == '__main__':
    # demo03()
    # demo04()
    # demo05()
    # demo06()
    demo07()


# 初始化权重
# torch.nn.init.normal_(tensor, mean=0, std=1)
# torch.nn.init.uniform_(tensor, a=0, b=1)
# torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
# torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
# torch.nn.init.xavier_uniform_(tensor, gain=1)
# torch.nn.init.xavier_normal_(tensor, gain=1)