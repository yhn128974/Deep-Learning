"""
演示
    自动微分模块 如何 计算梯度/导数
介绍：
    梯度：损失函数对模型参数的导数/梯度，针对某个确定的模型参数，损失函数的导函数的数值，模型参数就是权重w,偏置b
    损失函数对模型参数的梯度不需要手动计算，是loss.backward反向传播过程中pytorch调用自动微分模块来自动计算的
    梯度下降法的参数更新公式：
       w新=w旧-学习率*梯度（损失函数的导数）

过程：
    1.定义模型权重参数
    2.定义一个损失函数
    3.backward()反向传播计算导数/梯度
    4.w.grad获取导数值/梯度
    5.更新参数：w新 = w旧 - 学习率 * 梯度
 
注意：只有标量张量才能求导，且大多数底层操作都是浮点型，记得转换数据类型

"""





# import torch

# # 1.定义张量,记录初始的权重w
# # data记录张量的数据
# # requires_grad=True 告诉pytorch需要计算梯度
# # dtype=torch.float32 告诉pytorch张量的类型，必须是浮点型
# w=torch.tensor(data=10,requires_grad=True,dtype=torch.float32)

# # 定义loss变量，表示损失函数
# loss=2*w**2+10

# # 打印loss的梯度函数类型
# print(f"loss的梯度函数类型: {type(loss.grad_fn)}")

# # 计算梯度=损失函数的导数
# loss.backward()


# # 代入 权重更新公式:w新=w旧-学习率*梯度
# w.data=w.data-0.01*w.grad

# # 一次梯度下降的结果
# print(f"w的值为{w.data}")


import torch

w=torch.tensor(data=10,requires_grad=True,dtype=torch.float32)

loss=w**2+20

# 利用梯度下降方法，进行循环迭代
print(f"开始 权重的初始值：{w},(0.01*w.grad):无，loss:{loss})")

# 进行梯度迭代 range的前闭后开区间
for i in range(1,101):
    loss=w**2+20

    if w.grad is not None:
        w.grad.zero_()
    
    # 生成w.grad 进行反向传播
    loss.sum().backward()

    # 更新w.data的值
    w.data=w.data-0.01*w.grad

    print(f"第{i}次 权重的初始值：{w.data:.5f},(0.01*w.grad):{0.01*w.grad:.5f},loss:{loss:.5f})")


 




    


