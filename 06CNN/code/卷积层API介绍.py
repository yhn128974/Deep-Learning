"""
演示卷积层的API，用于提取图像的局部特征，并获取特征图

卷积神经网络：
    概述：
        convolutional neural network
    结构：
        卷积层：
            用于提取图像的局部特征结合卷积核（每个卷积核就是一个神经元（加权求和）），处理后的结果就是特征图
        池化层：
            用于降维，降采样
        全连接层：
            用于预测结果概率
    特征图大小计算方式：
        N=（W(输入大小)-F（卷积核）+2P（padding大小））/S（跨步）+1

"""


import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

current_dir=os.path.dirname(os.path.abspath(__file__))
image_url=os.path.join(current_dir,'../img/OIP.png')

def demo01():
    # PIL.Image.open 会根据文件二进制头自动识别真实格式（即使 .webp 被误命名为 .png 也能正常打开）
    img = np.array(Image.open(image_url))

    # print(f"img:\n{img}\nshape:{img.shape}")

    # numpy改变为tensor类型并进行数据格式转换
    img2=torch.tensor(img,dtype=torch.float)
    img2=img2.permute(2,0,1)
    # print(f"img2:\n{img2}\nshape:{img2.shape}")

    # 因为这里只有一张图，所以我们要给他进行升维重CHW->(1,C,H,W)一张三通达的像素图
    img3=img2.unsqueeze(dim=0)
    # print(f"img2:\n{img3}\nshape:{img3.shape}")


    # 创建卷积对象，提取特征图
    # 参数解释：
        # 参数1：输入通道数（层高|RGB三通道）
        # 参数2：输出通道数（卷积核个数|神经元个数）
        # 参数3：卷积核的大小
        # 参数4：步长
        # 参数5:填充PADDING
    conv=nn.Conv2d(3,4,3,2,0)

    # 卷积计算
    conv_img=conv(img3)

    # print(f'conv_img:{conv_img},shape:{conv_img.shape}') ([1, 4, 208, 139])
    # 可视化第一个通道特征图 
    img4=conv_img[0]

    # 吧CHW转化为HWC格式
    img5=img4.permute(1,2,0)

    # 可视化第一个元素 (获取4个特征图的第一张图)
    feature1=img5[:,:,1].detach().numpy()
    plt.imshow(feature1)
    plt.show()


if __name__=='__main__':
    demo01()