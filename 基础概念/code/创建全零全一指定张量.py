"""
演示
    创建 全0、全1、指定值 张量
涉及到的API:
    torch.zeros 和 torch.zeros_like 创建全0张量
    torch.ones 和 torch.ones_like 创建全1张量
    torch.full 和 torch.full_like 创建全为指定值张量

要掌握：
    torch.zeros
用处：
    网络参数初始化，偏置参数初始化
"""

import torch
import math
def demo01():
    # 全零
    t1=torch.zeros(3,4)
    print(f"{t1}, shape:{t1.shape}, dtype: {t1.dtype}, dim: {t1.ndim}")
    # 全一
    t2=torch.ones(3,4)
    print(f"{t2}, shape:{t2.shape}, dtype: {t2.dtype}, dim: {t2.ndim}")
    # 填满一个指定值
    t3=torch.full((3,4),math.pi)
    print(f"{t3}, shape:{t3.shape}, dtype: {t3.dtype}, dim: {t3.ndim}")


def demo02():
    # 使用一个已有的张量来创建相同大小的张量
    t1=torch.tensor([[1,2,3],[4,5,6]])
    t2=torch.zeros_like(t1)
    t3=torch.ones_like(t1)
    t4=torch.full_like(t1,7)
    print(f"t1:{t1}, shape:{t1.shape}, dtype: {t1.dtype}, dim: {t1.ndim}")
    print(f"t2:{t2}, shape:{t2.shape}, dtype: {t2.dtype}, dim: {t2.ndim}")
    print(f"t3:{t3}, shape:{t3.shape}, dtype: {t3.dtype}, dim: {t3.ndim}")
    print(f"t4:{t4}, shape:{t4.shape}, dtype: {t4.dtype}, dim: {t4.ndim}")

    
if __name__ == '__main__':
    # demo01()
    demo02()
    