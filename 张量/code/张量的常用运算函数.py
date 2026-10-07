"""
演示
    张量的常用运算函数,对同一个张量进行运算

涉及到的API:
    sum(), mean(), max(), min()         -> 有dim参数，可以指定维度/轴进行操作，会改变张量的形状
    pow()/**, sqrt()/**0.5, exp(), log(), log2(), log10() -> 无dim参数，对张量中的所有元素进行运算，不会改变张量的形状

要掌握的：
    sum(), mean(), max(), min()
    ** 求n次方
"""
# 导包
import torch

# 1.定义函数，演示 常用的运算函数
def demo01():
    # 0.设置随机种子
    torch.manual_seed(5)


    # 1.创建一个3D张量，(3,4,5)
    t1 = torch.randint(0,10,(3,4,5))
    print(f"t1: \n{t1}, shape: {t1.shape}, dtype: {t1.dtype}")
    print('=='*20)


    # 2.演示 sum(), mean(), max(), min()
    # 2.1 演示 sum()
    # 没有dim参数，则返回标量
    # print(f"t1.sum(): {t1.sum()}, shape: {t1.sum().shape}, ndim: {t1.sum().ndim}")
    # print(f"t1.sum(dim=0): \n{t1.sum(dim=0)}, shape: {t1.sum(dim=0).shape}, ndim: {t1.sum(dim=0).ndim}")
    # dim=0: (3,4,5) -> (4,5)
    # print(f"t1.sum(dim=1): \n{t1.sum(dim=1)}, shape: {t1.sum(dim=1).shape}, ndim: {t1.sum(dim=1).ndim}")
    # dim=1: (3,4,5) -> (3,5)

    # 2.2 演示 mean()
    t1 = t1.to(torch.float32)
    # print(f"t1.mean(): {t1.mean()}, shape: {t1.mean().shape}, ndim: {t1.mean().ndim}")
    # print(f"t1.mean(dim=0): {t1.mean(dim=0)}, shape: {t1.mean(dim=0).shape}, ndim: {t1.mean(dim=0).ndim}")
    # # (4,5)
    # print(f"t1.mean(dim=1): {t1.mean(dim=1)}, shape: {t1.mean(dim=1).shape}, ndim: {t1.mean(dim=1).ndim}")
    # # (3,5)
    # print(f"t1.mean(dim=2): {t1.mean(dim=2)}, shape: {t1.mean(dim=2).shape}, ndim: {t1.mean(dim=2).ndim}")
    # (3,4)

    


    # # 2.3 演示 max()
    # print(f"t1.max(): {t1.max()}, shape: {t1.max().shape}, ndim: {t1.max().ndim}")
    # print(f"t1.max(dim=0): {t1.max(dim=0)}, shape: {t1.max(dim=0)[0].shape}")
    # # (4,5)
    # print(f"t1.max(dim=1): {t1.max(dim=1)}, shape: {t1.max(dim=1)[0].shape}")
    # # (3,5)


    # # 2.4 演示 min()
    # print(f"t1.min(): {t1.min()}, shape: {t1.min().shape}, ndim: {t1.min().ndim}")
    # print(f"t1.min(dim=0): {t1.min(dim=0)}, shape: {t1.min(dim=0)[0].shape}")
    # # (4,5)
    # print(f"t1.min(dim=1): {t1.min(dim=1)}, shape: {t1.min(dim=1)[0].shape}")
    # (3,5)


    # 3.演示 pow()/**, sqrt(), exp(), log(), log2(), log10()
    # 3.1 演示 **
    # print(f"t1**0.5: {t1**0.5}, shape: {(t1**0.5).shape}, ndim: {(t1**0.5).ndim}")
    # (3,4,5)


    # # 3.2 演示 sqrt 开根号
    # print(f"t1.sqrt(): {t1.sqrt()}, shape: {(t1.sqrt()).shape}, ndim: {(t1.sqrt()).ndim}")
    ...
    #  # 3.3 演示 pow
    # print(f"t1 pow: {t1.pow(2) }, shape: {(t1.pow(2)).shape}, ndim: {(t1.pow(2)).ndim}")
    
    # exp() 以e为底进行指数运算
    # print(f"t1 exp: {t1.exp() }, shape: {(t1.exp()).shape}, ndim: {(t1.exp()).ndim}")


# 测试
if __name__ == '__main__':
    demo01()