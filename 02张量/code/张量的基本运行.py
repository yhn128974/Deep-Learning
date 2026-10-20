
"""
演示
    张量的基本运算，也就是 + - * / -
涉及到的API:
    add、sub、mul、div、neg
    add_、sub_、mul_、div_、neg_（其中带下划线的版本会修改原数据）,inplace=True
需要掌握：
    + - * /
注意：
    1.张量 和 标量的运算：就是张量的每个元素和这个标量进行运算
    2.pytorch中可以通用的API：Tensor.X = torch.X(Tensor,)

"""

# 导包
import torch

# 1.定义函数，演示 张量的基本运算
def demo01():
    # 0.设置随机种子
    torch.manual_seed(5)
    # 1.定义一个张量，(3,4)
    t1 = torch.randint(0,10,(3,4))
    # print(f"t1:{t1}, shape: {t1.shape}, dtype: {t1.dtype}")
    # 2.演示 +
    # t2 = t1 + 1
    # t2 = t1.add(1)
    # t2 = torch.add(t1,1)
    # t2 = t1.add_(1) # inplace=True

    # 3.演示 -
    # t2 = t1 - 1
    # t2 = t1.sub(1)
    # t2 = torch.sub(t1,1)
    # t2 = t1.sub_(1)

    # 4.演示 *
    # t2 = t1 * 2
    # t2 = t1.mul(2)
    # t2 = torch.mul(t1,2)
    # t2 = t1.mul_(2)

    # # 5.演示 /
    # t2 = t1 / 2
    # t2 = t1.div(2)
    # t2 = torch.div(t1,2)
    # t2 = t1.div_(2)

    # 6.演示 取负 -
    # t2 = - t1
    # t2 = t1.neg()
    # t2 = torch.neg(t1)
    # t2 = t1.neg_()

    print(f"t1:{t1}, shape: {t1.shape}, dtype: {t1.dtype}")
    print(f"t2:{t2}, shape: {t2.shape}, dtype: {t2.dtype}")
   

# 测试
if __name__ == '__main__':
    demo01()