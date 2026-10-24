
"""
池化不会改变通道数，只会改变长宽(降采样)
    最大池化
    平局池化
"""
import torch
import torch.nn as nn
import os
import numpy as np


# current_dir=os.path.dirname(os.path.abspath(__file__))
# image_url=os.path.join(current_dir, '../img/OIP.png')

def demo01():
    data=torch.randn(1,3,3)
    # print(f"data{data},shap:{data.shape}")

    # 最大池化
    # 建立池化对象
    pool1=nn.MaxPool2d(kernel_size=2,stride=1,padding=0)
    # 池化计算
    pool_data1=pool1(data)
    print(f"最大池化后：{pool_data1},shap:{pool_data1.shape}")

    #平均池化
    # 建立池化对象
    pool2=nn.AvgPool2d(kernel_size=2,stride=1,padding=0)
    # 池化计算
    pool_data2=pool2(data)
    print(f"平均池化后：{pool_data2},shap:{pool_data2.shape}")

def demo02():
    data=torch.tensor([
        [
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ],
        [
            [10,20,30],
            [40,50,60],
            [70,80,90]
        ],
        [
            [11,22,33],
            [44,55,66],
            [77,88,99]
        ]
    ]
    )
    # print(f"data{data},shap:{data.shape}")

    # 最大池化
    # 建立池化对象
    pool1=nn.MaxPool2d(kernel_size=2,stride=1,padding=0)
    # 池化计算
    pool_data1=pool1(data)
    print(f"最大池化后：{pool_data1},shap:{pool_data1.shape}")

    #平均池化
    # 建立池化对象
    pool2=nn.AvgPool2d(kernel_size=2,stride=1,padding=0)
    # 池化计算
    pool_data2=pool2(data)
    print(f"平均池化后：{pool_data2},shap:{pool_data2.shape}")
    
if __name__=='__main__':
    demo02()
    

