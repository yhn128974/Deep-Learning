import torch

# 定义x，表示特征数据
x=torch.ones(2,5)
print(f'x.data:{x.data}')

# 定义y，表示数据标签
y=torch.zeros(2,3)
print(f'y.data:{y.data}')

# 初始化权重和偏置
w=torch.randn(5,3,requires_grad=True)
 
b=torch.randn(3,requires_grad=True)

# 向前传播（正向传播）,计算出 预测值（z）
y_pred=torch.matmul(x,w)+b
# y_pred=x@w+b
print(f"y_pred:{y_pred}")


# 定义损失函数对象
criterion=torch.nn.MSELoss()
# 计算 损失值
loss=criterion(y_pred,y)
print(f"loss:{loss}")


# 进行自动微分，求导结合反向传播，更新权重!!!
loss.backward()

#打印w,b用来更新的梯度
print(f"w.grad:{w.grad}")
print(f"b.grad:{b.grad}")

# 更新权重为偏执 w(i+1)=w-w*w梯度 
w=w-0.01*w.grad
b=b-0.01*b.grad


print(f"w更新后:{w}")
print(f"b更新后:{b}")

