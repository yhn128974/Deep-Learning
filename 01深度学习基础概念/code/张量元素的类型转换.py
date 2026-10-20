"""
演示
    张量元素类型的转换

设计到的API:
    data.to(torch.float32)
    data.type(torch.DoubleTensor)
    data.half/double/float/short/int/long()
需要掌握：
    data.to(dtype,device)
    data.type(dtype)

"""

import torch

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def demo01():
    t1=torch.tensor([1,2,3],dtype=torch.int32)
    t1=t1.to(device)
    print(f"t1:{t1}, shape:{t1.shape}, dtype: {t1.dtype}, dim: {t1.ndim}, device: {t1.device}")

    # t2=t1.to(dtype=torch.float32,device='cpu')
    # t2=t1.to(torch.float32)
    # t2=t1.float()
    t2=t1.type(torch.float)
    t2=t2.to(device) 
    print(f"t2:{t2}, shape:{t2.shape}, dtype: {t2.dtype}, dim: {t2.ndim}, device: {t2.device}")



if __name__=='__main__':
    demo01()




