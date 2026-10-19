"""
演示
    常用的激活函数
人工神经元组成：
    加权求和 + 激活函数(引入非线性)
sigmoid
    用于二分类任务的输出层，通常用于浅层网络，不超过5层，容易出现梯度消失，目前使用很少
    输出范围(0,1),导数范围(0,0.25]
ReLU
    常用于 隐藏层，目前使用最多.
    计算公式max(0,x),输出范围[0,+∞),导数{0,1}
    优势：
        1.计算简单，模型收敛快
        2.正输入的导数恒为1，缓解了梯度消失问题
    DeadReLU问题：
        随着训练的推进，部分输入会落入负区域，恒为负,模型输出恒为0,导致对应权重无法更新。被称为“神经元死亡”。大模型训练时，通常采用GELU。

tanh
    用于隐藏层，浅层网络，同样存在梯度消失问题，很少使用，主流网络中只有RNN(LSTM,GRU)使用了tanh
    输出范围(-1,1),导数范围(0,1]

softmax
    用于单标签多分类任务的输出层，二分类任务上可以替代sigmoid.
    输出各个类别的概率分布，范围(0,1), 并且概率总和为1.
    
激活函数的选择方法
    隐藏层:	
        ReLU -> Leaky ReLU/PReLU/GELU -> (Tanh)

    输出层:
        多分类问题选择 softmax
        二分类问题选择 sigmoid 或 softmax
        回归问题多数选择 identity (就是没有激活函数)，正数ReLU，区间Sigmoid/Tanh

需要掌握的：
    sigmoid公式、应用(面试)
    torch.relu
    torch.softmax

bug解决：
    绘制激活函数图像时出现以下提示:
    OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    解决方法:
        方法1：更换版本matplotlib, pip install matplotlib==3.7.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
        方法2：更换版本numpy, pip install numpy==1.26.3 -i https://pypi.tuna.tsinghua.edu.cn/simple

"""

# 导包
import torch    # pytorch框架
import torch.nn.functional as F # 函数模块
import matplotlib.pyplot as plt
import torch.nn as nn   # 神经网络模块，提供各种网络层

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False            # 解决负号显示问题

# 1.定义函数，演示sigmoid激活函数的函数图像和导函数图像
def demo01():
    # 1.创建画布
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # 2.生成x值，-10到10之间，总数量100
    x = torch.linspace(-10, 10, 100)
    # 3.计算y值，sigmoid激活函数
    y = torch.sigmoid(x)
    # y = F.sigmoid(x)
    # y = nn.Sigmoid()(x)
    # 4.绘制函数图像，在第一个子图上
    axes[0].plot(x, y)
    axes[0].set_title('sigmoid激活函数')
    axes[0].grid()
    # 5.绘制导函数图像，在第二个子图上
    x = torch.linspace(-10, 10, 100,requires_grad=True) # 1D:(100,)
    y = torch.sigmoid(x) # 1D:(100,)
    # y = F.sigmoid(x)
    # y = nn.Sigmoid()(x)
    y.sum().backward()  # 方向传播获取导数，x.grad
    # 如果.mean()就是先求和，再除以元素数量，会变成导数的1/100
    axes[1].plot(x.detach(), x.grad)
    axes[1].set_title('sigmoid激活函数导数')
    axes[1].grid()
    plt.show()
    ...

# 2.定义函数，演示relu激活函数的函数图像和导函数图像
def demo02():
    # 1.创建画布
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # 2.生成x值，-10到10之间，总数量100
    x = torch.linspace(-10, 10, 1000)
    # 3.计算y值，relu激活函数
    y = torch.relu(x)
    # y = F.relu(x)
    # y = nn.ReLU()(x)
    # 4.绘制函数图像，在第一个子图上
    axes[0].plot(x, y)
    axes[0].set_title('relu激活函数')
    axes[0].grid()
    # 5.绘制导函数图像，在第二个子图上
    x = torch.linspace(-10, 10, 1000,requires_grad=True)
    y = torch.relu(x)
    y.sum().backward()  # 方向传播获取导数，x.grad
    axes[1].plot(x.detach(), x.grad)
    axes[1].set_title('relu激活函数导数')
    axes[1].grid()
    plt.show()
    ...

# 3.定义函数，演示tanh激活函数的函数图像和导函数图像
def demo03():
    # 1.创建画布
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # 2.生成x值，-10到10之间，总数量100
    x = torch.linspace(-10, 10, 1000)
    # 3.计算y值，tanh激活函数
    y = torch.tanh(x)
    # 4.绘制函数图像，在第一个子图上
    axes[0].plot(x, y)
    axes[0].set_title('tanh激活函数')
    axes[0].grid()
    # 5.绘制导函数图像，在第二个子图上
    x = torch.linspace(-10, 10, 1000,requires_grad=True)
    y = torch.tanh(x)
    y.sum().backward()  # 方向传播获取导数，x.grad
    axes[1].plot(x.detach(), x.grad)
    axes[1].set_title('tanh激活函数导数')
    axes[1].grid()
    plt.show()
    ...

# 4.定义函数，演示softmax激活函数的函数图像和softmax(softmax)图像
def demo04():
    # 1.创建画布
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 2.定义x,模拟模型的预测分数logits
    x = torch.tensor([0.2,-0.9,1,-10,1,2,3,4,2.5,3.9])    # (10,)
    # 举例说明：老板请客，你可以点一个菜，菜单上有10个菜，选了2对应的菜，最大预测分数10对应的类别
    # 3.经过softmax运算
    y = torch.softmax(x, dim=-1) # 通常特征维度/轴为-1轴
    # y = F.softmax(x, dim=-1)
    # 4.绘制softmax函数图像，在第一个子图上
    axes[0].bar(range(len(y)), y)
    axes[0].set_title('softmax激活函数')
    axes[0].grid()
    # 5.绘制softmax(softmax)函数图像，在第二个子图上
    z = torch.softmax(y, dim=-1)
    axes[1].bar(range(len(z)), z)
    axes[1].set_title('softmax(softmax)激活函数')
    axes[1].grid()
    plt.show()
    ...

# 5.定义函数，演示gelu激活函数的函数图像和导函数图像
def demo05():
    # 1.创建画布
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # 2.生成x值，-10到10之间，总数量100
    x = torch.linspace(-10, 10, 1000)
    # 3.计算y值，gelu激活函数
    # y = torch.gelu(x)
    # y = F.gelu(x)
    y = nn.GELU()(x)
    # 4.绘制函数图像，在第一个子图上
    axes[0].plot(x, y)
    axes[0].set_title('gelu激活函数')
    axes[0].grid()
    # 5.绘制导函数图像，在第二个子图上
    x = torch.linspace(-10, 10, 1000,requires_grad=True)
    # y = torch.gelu(x)
    # y = F.gelu(x)
    y = nn.GELU()(x)
    y.sum().backward()  # 方向传播获取导数，x.grad
    axes[1].plot(x.detach(), x.grad)
    axes[1].set_title('gelu激活函数导数')
    axes[1].grid()
    plt.show()
    ...

# 测试
if __name__ == '__main__':
    # demo01()
    # demo02()
    # demo03()
    demo04()
    # demo05()