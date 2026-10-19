"""
演示
    梯度下降法 求损失函数的最小值
    循环实现，计算梯度，更新参数w
题目：
    # 求 loss = w**2 + 20 的极小值点 并打印loss是最小值时 w的值(梯度)
计算过程：
    # 1 定义点 w=10, requires_grad=True, dtype=torch.float32
    # 2 定义函数 loss = w**2 + 20
    # 3 利用梯度下降法 循环迭代 求最优解/最小值
        # 3.1 前向传播,计算损失
        # 3.2 梯度清零 w.grad.zero_()
        # 3.3 反向传播 loss.backward()
        # 3.4 更新参数 w.data = w.data - 0.01 * w.grad
"""

# 导包
import torch
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False            # 解决负号显示问题

# 1 定义点 w=10, requires_grad=True, dtype=torch.float32
w = torch.tensor([10.0],requires_grad=True,dtype=torch.float32)
# 2 定义函数 loss = w**2 + 20
loss = w**2 + 20    # 120.0
# 3 利用梯度下降法 循环迭代 求最优解/最小值
print(f"初始值：w:{w.data}, w.grad: {w.grad}, loss.mean():{loss.mean()}")
# 记录损失均值
loss_list = []
for i in range(500):
    # 3.1 前向传播, 计算损失
    loss = w**2 + 20
    # 3.2 梯度清零 w.grad.zero_()
    if w.grad is not None:
        w.grad.zero_()
    # 3.3 反向传播 loss.backward()
    loss.mean().backward()
    print(f"w.grad:{w.grad.data}") 
    # loss_mean' = 2*w
    # 3.4 更新参数 w.data = w.data - 0.01 * w.grad
    w.data = w.data - 0.01 * w.grad.data
    # loss_mean' = 2*w = w = [2*w1]
    print(f"第{i+1}次迭代：w:{w.data}, w.grad:{w.grad.data}, loss.mean():{loss.mean()}")
    # 添加到loss_list
    loss_list.append(loss.mean().item())

# 绘制损失曲线
iterations = range(500)
plt.figure(figsize=(12,8))
plt.plot(iterations,loss_list)
plt.xlabel("迭代次数")
plt.ylabel("损失均值loss.mean")
plt.grid()
plt.show()