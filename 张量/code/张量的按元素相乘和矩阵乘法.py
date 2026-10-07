"""
演示
    按元素相乘 和 矩阵乘法，对两个张量进行运算
按元素相乘:
    按元素相乘（Hadamard）指的是两个相同形状的张量对应位置的元素相乘，使用 mul和运算符 * 实现
    要求：两个张量的形状一样，A:(m,n),B(m,n)
    API:
        t1*t2
        t1.mul(t2)
        torch.mul(t1,t2)

矩阵乘法：
    要求：A(m,n),B必须是(n,p),也就是第一个矩阵A.shape[-1]=B.shape[-2]
        如果是3D及以上张量，还要求A.shape[:-2]=B.shape[:-2]
    运算：A的行向量 和 B的列向量 做点积，也就是先做按元素相乘再求和
        (m,n)@(n,p)=(m,p)
    API:
        t1@t2
        t1.matmul(t2)
        torch.matmul(t1,t2)
        t1.dot(t2),用于1D张量/向量的点积运算

需要掌握：
    t1*t2
    t1@t2

"""

# 导包
import torch

# 1.定义函数，演示 张量的按元素位置在原位置相乘
def demo01():
    # 0.设置随机种子
    torch.manual_seed(5)
    # 1.创建两个张量，(3,4)
    t1 = torch.randint(0,10,(3,4))
    print(f"t1:\n{t1}, shape: {t1.shape}, dtype: {t1.dtype}")
    t2 = torch.randint(0,10,(3,4))
    print(f"t2:\n{t2}, shape: {t2.shape}, dtype: {t2.dtype}")
    # 2. 演示 按元素相乘
    t3 = t1*t2
    # t3 = t1.mul(t2)
    # t3 = torch.mul(t1,t2)
    print(f"t3:\n{t3}, shape: {t3.shape}, dtype: {t3.dtype}")


# 2.定义函数，演示 张量的矩阵乘法
def demo02():
    # 0.设置随机种子
    torch.manual_seed(5)
    # 1.创建两个张量，(3,4)@(4,3)=(3,3)
    t1 = torch.randint(0,10,(3,4))
    print(f"t1:\n{t1}, shape: {t1.shape}, dtype: {t1.dtype}")
    t2 = torch.randint(0,10,(4,5))
    print(f"t2:\n{t2}, shape: {t2.shape}, dtype: {t2.dtype}")
    # 2. 演示 矩阵乘法
    # t3 = t1@t2
    # t3 = t1.matmul(t2)
    t3 = torch.matmul(t1,t2)
    print(f"t3:\n{t3}, shape: {t3.shape}, dtype: {t3.dtype}")
    # 3.演示 点积dot
    print(f"t1[0,:3]:\n{t1[0,:3]}")
    print(f"t2[0,:3]:\n{t2[0,:3]}")
    # t4 = t1[0,:3].dot(t2[0,:3])
    t4 = torch.dot(t1[0,:3],t2[0,:3])
    print(f"t4:\n{t4}, ndim: {t4.ndim}, shape: {t4.shape}, dtype: {t4.dtype}")
    ...

# 测试
if __name__ == '__main__':
    # demo01()
    demo02()
