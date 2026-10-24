
# 输出图像的大小与原图像，padding，卷积核与步长都有关

import os
import numpy as np
import matplotlib.pyplot as plt
import torch


def demo01():
    img1=np.zeros((200,200,3))
    plt.imshow(img1)
    plt.show()


def demo02():
    img2=np.ones((200,200,3))
    plt.imshow(img2)
    plt.show()

def demo03():
    # 获取当前脚本所在目录，确保无论在哪个工作目录下执行都能正确定位图片
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_dir, '../img/截图.png')
    save_path = os.path.join(current_dir, '../img/截图_copy.png')

    img1 = plt.imread(img_path)
    plt.imsave(save_path, img1)
    

     # demo01()
    # demo02()
    demo03()


