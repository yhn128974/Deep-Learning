"""
演示
    学习率衰减策略/学习率调度策略
介绍：
    之前学的优化器，可以调节学习率：Adam/AdamW, AdaGrad, RMSProp,
    Momentum(调节梯度)
    也可以手动调节学习率：
        等间隔学习率衰减
        指定间隔学习率衰减
        指数间隔学习率衰减
        周期重启的余弦退火策略
等间隔学习率衰减
    固定间隔来衰减学习率，当前学习率lr = 上一次的学习率lr*γ, 比如γ=0.5

指定间隔学习率衰减
    人为指定间隔来衰减学习率，当前学习率lr = 上一次的学习率lr*γ, 比如γ=0.5

指数间隔学习率衰减
    当前轮次的lr = 上个轮次lr*γ, 比如γ=0.95

周期重启的余弦退火策略
    带有周期重启的余弦曲线，可以让学习率重新启动，有可能冲过局部最小值

神经网络的训练流程：
    1.准备数据集
    2.构建神经网络模型
    3.设置损失函数和优化器，以及学习率调度器
    4.模型训练
        1.前向传播
        2.计算损失
        3.梯度清零
        4.反向传播
        5.更新参数:w新 = w旧-学习率*梯度
        6.更新学习率
    5.模型测试

需要掌握：
    无
    建议周期重启的余弦退火策略 optim.lr_scheduler.CosineAnnealingWarmRestarts

"""
import torch
from torch import optim
import matplotlib.pyplot as plt

# 等间隔学习率
def demo01():
    lr,epochs,iterations_per_epoch = 0.1,200,10
    # 真实值
    y_true=torch.tensor([0])
    # 输入值
    x=torch.tensor([1.0],dtype=torch.float32)
    # 参数w
    w=torch.tensor([1.0],requires_grad=True,dtype=torch.float32)
    #优化器参数：参数一：待训练的参数；参数二：学习率；参数三：动量
    optimizer=optim.SGD([w],lr=lr,momentum=0.9)
    # 创建等间隔学习率衰减器：参数一：优化器；参数二：步长；参数三：衰减率
    step_lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.5)

    # 创建两个列表，分别表示训练轮数，每轮训练的学习率
    epoch_list,lr_list = [],[]
    # 
    for epoch in range(epochs):

        epoch_list.append(epoch)

        lr_list.append(step_lr_scheduler.get_last_lr()[0])  #get_last_l获取上一次的学习率

        # 训练参数安装批次训练
        for batch in range(iterations_per_epoch):


            y_pred=w*x

            loss=(y_pred-y_true)**2
            # 归零
            optimizer.zero_grad()
            # 求导
            loss.backward()
            # 更新
            optimizer.step()

        # 更新学习率
        step_lr_scheduler.step()

   
    #打印结果
    print(f'step_lr_scheduler:{step_lr_scheduler}')


    # 可视化
    plt.plot(epoch_list,lr_list,label='StepLR (step=50, gamma=0.5)')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Decay')
    plt.legend()
    plt.show()

# 指定间隔学习率
def demo02():
    lr,epochs,iterations_per_epoch = 0.1,200,10
    # 真实值
    y_true=torch.tensor([0])
    # 输入值
    x=torch.tensor([1.0],dtype=torch.float32)
    # 参数w
    w=torch.tensor([1.0],requires_grad=True,dtype=torch.float32)

    #优化器参数：参数一：待训练的权重参数；参数二：学习率；参数三：动量
    optimizer=optim.SGD([w],lr=lr,momentum=0.9)

    #学习率衰减器（指定间隔学习率）：参数一：优化器；参数二：固定更新点；参数三：衰减率
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50,120,160],gamma=0.5)

    # 创建两个列表，分别表示训练轮数，每轮训练的学习率
    epoch_list,lr_list = [],[]

    # 
    for epoch in range(epochs):

        epoch_list.append(epoch)

        lr_list.append(scheduler.get_last_lr()[0])  #get_last_l获取上一次的学习率

        # 训练参数安装批次训练
        for batch in range(iterations_per_epoch):

            y_pred=w*x

            loss=(y_pred-y_true)**2
            # 优化器导数归零
            optimizer.zero_grad()
            # 损失函数求导
            loss.backward()
            # 优化器更新权重
            optimizer.step()

        # 更新学习率
        scheduler.step()

   
    #打印结果
    print(f'scheduler:{scheduler}')


    # 可视化
    plt.plot(epoch_list,lr_list,label='StepLR (step=50, gamma=0.5)')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Decay')
    plt.legend()
    plt.show()

# 指数间隔学习率衰减(更符合梯度下降规律)
def demo03():
    lr,epochs,iterations_per_epoch = 0.1,200,10
    # 真实值
    y_true=torch.tensor([0])
    # 输入值
    x=torch.tensor([1.0],dtype=torch.float32)
    # 参数w
    w=torch.tensor([1.0],requires_grad=True,dtype=torch.float32)

    #优化器参数：参数一：待训练的权重参数；参数二：学习率；参数三：动量
    optimizer=optim.SGD([w],lr=lr,momentum=0.9)

    #指数间隔学习率衰减(更符合梯度下降规律) ：参数一：优化器；参数二：固定更新点；参数三：衰减率
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.95)

    # 创建两个列表，分别表示训练轮数，每轮训练的学习率
    epoch_list,lr_list = [],[]

    # 
    for epoch in range(epochs):

        epoch_list.append(epoch)

        lr_list.append(scheduler.get_last_lr()[0])  #get_last_l获取上一次的学习率

        # 训练参数安装批次训练
        for batch in range(iterations_per_epoch):

            y_pred=w*x

            loss=(y_pred-y_true)**2
            # 优化器导数归零
            optimizer.zero_grad()
            # 损失函数求导
            loss.backward()
            # 优化器更新权重
            optimizer.step()

        # 更新学习率
        scheduler.step()
   
    #打印结果
    print(f'scheduler:{scheduler}')

    # 可视化
    plt.plot(epoch_list,lr_list,label='StepLR (step=50, gamma=0.5)')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Decay')
    plt.legend()
    plt.show()

# 周期重启的余弦退火策略

if __name__=='__main__':
    # demo01()
    # demo02()
    demo03()
