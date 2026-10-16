"""
二分类损失函数：loss=-y log(y_hat) - (1-y)log(1-y_hat)
y_hat: 预测概率(激活值)，由 sigmoid(y_logits) 得到
"""
import torch
import torch.nn as nn
def demo01():
    y_true=torch.tensor([1,0,1],dtype=torch.float32)
    # 预测值,通常由sigmoid输出的概率得到
    y_pred=torch.tensor([0.9,0.2,0.8],dtype=torch.float32)
    # 二分类交叉熵损失函数 内部没有自带激活函数需要输入的预测值是概率(经过了sigmoid)
    criterion=nn.BCELoss()
    loss=criterion(y_pred,y_true)
    print(f'二分类损失函数：{loss}')

if __name__ == "__main__":
    demo01()
    