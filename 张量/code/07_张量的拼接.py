"""
演示
    张量的拼接
涉及到的API:
    torch.cat()
        在指定的维度/轴拼接张量，不改变张量的维度2D/3D，会改变拼接的维度/轴的长度，除了要拼接的维度，其他维度形状必须相同
        (m,n)+(k,n)->(m+k,n)
    torch.stack()
        在一个新的维度/轴堆叠张量，增加一个新的维度，比如2D->3D,要求输入的张量的形状必须相同
        (m,n)+(m,n)->(2,m,n)

需要掌握的：
    torch.cat
    torch.stack

"""

# 导包
import torch

# 1.定义函数，演示 张量的拼接
def demo01():
    # 0.设置随机种子
    torch.manual_seed(28)
    # 1.定义两个张量，(3,4),(3,4)
    t1 = torch.randint(0,10,(3,4))
    t2 = torch.randint(0,10,(3,4))
    # t4 = torch.randint(0,10,(5,4))
    # print(f"t1:{t1}, shape: {t1.shape}")
    # print(f"t2:{t2}, shape: {t2.shape}")
    # 2.使用torch.cat(),拼接t1,t2
    # t3 = torch.cat([t1,t2],dim=0)  # (6,4)
    # print(f"t3:{t3}, shape: {t3.shape}")
    # t3 = torch.cat([t1, t2], dim=1)  # (3,8)

    # print(f"t3:{t3}, shape: {t3.shape}")
    # t3 = torch.cat([t1, t2], dim=2)  # (5,7)
    # print(f"t3:{t3}, shape: {t3.shape}")
     
    # 3.使用torch.stack(),堆叠t1,t2
    # t4 = torch.stack([t1, t2], dim=0)   # (2,3,4)
    # print(f"t4:{t4}, shape: {t4.shape}")
    # t4 = torch.stack([t1, t2], dim=1)  # (3,2,4)
    # print(f"t4:{t4}, shape: {t4.shape}")
    # t4 = torch.stack([t1, t2], dim=2)  # (3,4,2)
    # print(f"t4:{t4}, shape: {t4.shape}")
    ...

# 测试
if __name__ == '__main__':
    demo01()