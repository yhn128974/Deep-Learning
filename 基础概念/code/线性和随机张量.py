"""
演示
    创建 线性 和 随机张量
设计到的API:
    torch.arange() 和 torch.linspace() 创建线性张量
    torch.manual_seed() 设置随机种子
    torch.rand/randn() 创建随机浮点类型张量
    torch.randint(low, high, size=()) 创建随机整数类型张量

需要掌握的：
    torch.arange()
    torch.linspace()
    torch.manual_seed()
    torch.randint(low, high, size=())

"""

import torch

def demo01():
    t1=torch.arange(1,10,1)
    print(f't1:{t1},shape:{t1.shape},dtype:{t1.dtype},dim:{t1.ndim}')
    # 
    t2=torch.linspace(1,10,12,dtype=torch.float32)
    print(f't2:{t2},shape:{t2.shape},dtype:{t2.dtype},dim:{t2.ndim}')
    
def demo02():
    torch.manual_seed(0)
    # 生成范围为0-1
    t1=torch.rand(3,5)
    print(f"t1:\n{t1},shape:{t1.shape},dtype:{t1.dtype},dim:{t1.ndim}")
    print('='*30)
    torch.manual_seed(0)

    # 生成标准正态分布
    t2=torch.randn(3,5)
    print(f"t2:\n{t2},shape:{t2.shape},dtype:{t2.dtype},dim:{t2.ndim}")
    print('='*30)
    torch.manual_seed(0)

    # 生成指定范围的整数
    t3=torch.randint(0,10,size=(3,5))
    print(f"t3:\n{t3},shape:{t3.shape},dtype:{t3.dtype},dim:{t3.ndim}")
    print('='*30)
    torch.manual_seed(0)



if __name__ == '__main__':
    demo01()
    demo02()