"""
案例：
    演示 张量的索引操作
涉及到的API:
    简单行列索引
    列表索引
    范围索引
    多维索引
    布尔索引
需要掌握的：
    简单行列索引 t1[0,0], t1[0]
    范围索引 t1[:2], t1[:2,2:4], 起始(默认0):结束(默认最后):步长(默认1)

"""

# 导包
import torch

# 1.定义函数，演示 张量的索引操作
def demo01():
    # 0.设置随机种子
    torch.manual_seed(5)
    # 1.定义一个张量，(3,4)
    t1 = torch.randint(0,10,(3,4))
    print(f"t1:\n{t1}, shape: {t1.shape}, dtype: {t1.dtype}")

    # 2.演示 简单行列索引
    # print(f"t1[0,0]: {t1[0,0]}, ndim: {t1[0,0].ndim},shape: {t1[0,0].shape}")
    # print(f"t1[1]: {t1[1]},ndim: {t1[1].ndim},shape: {t1[1].shape}")
    # print(f"t1[:,2]: {t1[:,2]},ndim: {t1[:,2].ndim},shape: {t1[:,2].shape}")

    # 3.演示 列表索引
    # 列表索引就是把索引变成列表，然后对列表进行索引，[[x1,y1],[x2,y2]]=>x1,x2是第一个元素，y1,y2是第二个元素;
    # 前面的表示行，后面的表示列，对应位置匹配
    # # 获取(0,1),(1,2)位置的元素
    # print(f"t1[[0,1],[1,2]]: {t1[[0,1],[1,2]]}, ndim: {t1[[0,1],[1,2]].ndim},shape: {t1[[0,1],[1,2]].shape}")
    # # 获取(0,2),(1,3)位置的元素
    # print(f"t1[[0,1],[2,3]]: {t1[[0,1],[2,3]]}, ndim: {t1[[0,1],[2,3]].ndim},shape: {t1[[0,1],[2,3]].shape}")
    # 获取第一行到第二行的第三列到第四列
    # print(f"{t1[[[0],[1]],[2,3]]}, ndim: {t1[[[0],[1]],[2,3]].ndim},shape: {t1[[[0],[1]],[2,3]].shape}")




    # 4.演示 范围索引
    # # 获取前两行
    # print(f"t1[:2]: {t1[:2]}, ndim: {t1[:2].ndim},shape: {t1[:2].shape}")
    # # 获取前两列
    # print(f"t1[:,:2]: {t1[:,:2]}, ndim: {t1[:,:2].ndim},shape: {t1[:,:2].shape}")
    # # 获取偶数行
    # t2 = t1[1::2]
    # print(f"t2: {t2}, ndim: {t2.ndim},shape: {t2.shape}")
    # # 获取偶数行的奇数列
    # t3 = t1[1::2, ::2]
    # print(f"t3: {t3}, ndim: {t3.ndim},shape: {t3.shape}")

    # 5.演示 多维索引
    # t4 = torch.randint(0,10,(3,4,5))
    # print(f"t4:\n{t4}, shape: {t4.shape}, dtype: {t4.dtype}")
    # # 获取(0,0,0)位置的元素
    # t5 = t4[0,0,0]
    # print(f"t5: {t5}, ndim: {t5.ndim},shape: {t5.shape}")
    # # 获取0轴索引1，1轴偶数索引，2轴前3个 的数据
    # t6 = t4[1,::2,:3]   # (2,3)
    # print(f"t6: {t6}, ndim: {t6.ndim},shape: {t6.shape}")
    
    # 6.演示 布尔索引
    # 获取第二行的大于5的所有元素
    # print(t4[1]>5)  # (3,5)
   
    # print(f"t5: {t5}, ndim: {t5.ndim},shape: {t5.shape}")
    ...

# 测试
if __name__ == '__main__':
    demo01()
