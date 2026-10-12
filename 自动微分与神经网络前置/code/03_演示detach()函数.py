"""
演示
    detach() 函数的功能，解决 开启梯度计算 requires_grad=True 的张量，无法转为numpy对象的问题
介绍：
    一个张量Tensor，开启了梯度计算requires_grad=True, 就无法Tensor.numpy(),需要Tensor.detach().numpy()来转为numpy数组
需要掌握
    n1 = t1.detach().numpy()

"""

# 导包
import torch

# 1.定义张量，开启梯度计算requires_grad = True
w = torch.tensor([1.0,2.0,3.0], requires_grad=True, dtype=torch.float32)
print(f"w:{w.data}, shape: {w.shape}, requires_grad: {w.requires_grad}")

# # 2.直接转为numpy数组
# n1 = w.numpy()
# print(f"n1:{n1}, shape: {n1.shape}")

# 3.使用.detach().numpy()来将torch数组转为numpy数组
n2 = w.detach().numpy()
print(f"n2:{n2}, shape: {n2.shape}")

# # 4.查看是否共享内存/浅拷贝
n2[0]=5.0
print("="*30)
print(f"w:{w}, shape: {w.shape}")
print(f"n2:{n2}, shape: {n2.shape}")



