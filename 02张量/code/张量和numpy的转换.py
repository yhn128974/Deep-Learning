"""
演示
    张量 和 numpy数组 之间的相互转换，以及 只有一个数值的张量 转换为 python数值

涉及到的API:
    张量 转 numpy数组对象：
        张量对象.numpy()      共享内存，浅拷贝，两个变量一起变
        张量对象.numpy().copy() 不共享内存，深拷贝，两个变量不一起变
    numpy数组对象 转 张量：
        torch.from_numpy(numpy数组对象)  共享内存，浅拷贝
        torch.tensor(numpy数组对象)       不共享内存，深拷贝
    一个数值的张量 转 python数值：
        张量对象.item()

需要掌握：
    张量 转 numpy数组对象：
        张量对象.numpy()
    numpy数组对象 转 张量：
        torch.tensor(numpy数组对象)
    一个数值的张量 转 python数值：
        张量对象.item()

"""

# 导包
import torch
import numpy as np
# device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1.定义函数，演示 张量 和 numpy数组之间 的相互转换
def demo01():
    torch.manual_seed(254)
    # torch.randn 生成的是标准正态分布随机数，正态分布是连续分布，底层仅支持浮点类型
    t1=torch.randn(3,4)
    # 自动类型转换
    t1=t1.to(dtype=torch.float)
    # 打印张量的基本信息
    print(f"t1:{t1},ndim: {t1.ndim},shape: {t1.shape}, type: {type(t1)}, dtype: {t1.dtype}")
    print("=="*40)

    # 共享内存：两个变量一起变
    n1 = t1.numpy()
    n1[0,0]=100
    print(f"n1:{n1},ndim: {n1.ndim},sh1ape: {n1.shape},type: {type(n1)}, dtype: {n1.dtype}")
    print(t1)
    print("=="*40)


    # 不共享内存：两个变量不一起变
    n2=t1.numpy().copy()    
    n2[0,0]=200
    print(f"n2:{n2},ndim: {n2.ndim},shape: {n2.shape},type: {type(n2)}, dtype: {n2.dtype}")
    print(t1)
    print("=="*40)

    #torch from numpy 共享内存
    t2=torch.from_numpy(n2)
    t2[0,0]=300
    print(f"t2:{t2},ndim: {t2.ndim},shape: {t2.shape},type: {type(t2)}, dtype: {t2.dtype}")
    print(n2)
    print("=="*40) 

    # torch tensor 不共享内存
    t3=torch.tensor(n2)
    t3[0,0]=400
    print(f"t3:{t3},ndim: {t3.ndim},shape: {t3.shape},type: {type(t3)}, dtype: {t3.dtype}")
    print(n2)
    print("=="*40)

    #将张量转化为基本数值
    t4=torch.tensor([10])
    v4=t4.item()
    print(f"v4:{v4},type: {type(v4)}")
    print(t4)




 


# 测试
if __name__ == '__main__':
    demo01()