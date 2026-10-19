import torch

def demo01():
    t1=torch.tensor(1.00)
    print(f"t1:{t1}, shape: {t1.shape}, dtype: {t1.dtype},dim:{t1.ndim}")

def demo02():
    t2=torch.tensor([1.00,2.00,3.00],dtype=torch.float32)
    print(f"t2:{t2}, shape: {t2.shape}, dtype: {t2.dtype},dim:{t2.ndim}")

def demo03():
    t3=torch.tensor([[1.00,2.00,3.00],[4.00,5.00,6.00]], dtype=torch.int32)
    print(f"t3:{t3}, shape: {t3.shape}, dtype: {t3.dtype},dim:{t3.ndim}")

def demo04():
    # 两层两行三列
    t4=torch.tensor([[[1.00,2.00,3.00],[4.00,5.00,6.00]],[[7.00,8.00,9.00],[10.00,11.00,12.00]]])
    print(f"t4:{t4}, shape: {t4.shape}, dtype: {t4.dtype},dim:{t4.ndim}")

def demo05():
    c0=torch.Tensor(4,3)
    print(f'c0: {c0},shape: {c0.shape}')
    
    c1=torch.IntTensor(4,3)
    print(f'c1: {c1},shape: {c1.shape}')

    c2=torch.FloatTensor(4,3)
    print(f'c2: {c2},shape: {c2.shape}')

    c3=torch.DoubleTensor(4,3)
    print(f'c3: {c3},shape: {c3.shape}')
    
if __name__ == "__main__":
    # demo01()
    demo02()
    demo03()
    # demo04()
    # demo05()



    