"""
演示
    多分类交叉熵损失函数

介绍
    损失函数 也叫 成本函数/目标函数/代价函数/误差函数
    用来 衡量预测值和真实值的差距，从而衡量模型好坏，损失越小模型越好

多分类交叉熵损失
    公式：CrossEntropLoss = -Σy*log(softmax(y_logits))
    其中：y_logits是模型的原始输出|预测分数
        softmax(y_logits)是模型输出的概率分布，也就是每个类别的预测概率，最大概率对应预测类别(激活值)
    注意：
        多分类交叉熵CrossEntropLoss里面有softmax运算，所以模型的输出层不需要softmax。
        但是如果要获取预测概率(激活值)，则需要在模型输出的原始预测分数后面经过softmax
    应用：
        多分类问题，如：10个类别，预测图片属于哪一类
    API:
        nn.CrossEntropyLoss()

需要掌握：
    loss_fn = nn.CrossEntropLoss()
    loss = loss_fn(y_logits, y_true)

"""

import torch
import torch.nn as nn

def demo01():
    # 样本的真实值 one-hot编码
    y_true=torch.tensor([[1,0,0],[0,1,0]],dtype=torch.float32)
    # 模型的预测分数() 还没有经过softmax
    y_logits = torch.tensor([[0.1,0.8,0.1],[0.4,0.5,0.1]],dtype=torch.float32,requires_grad=True)
    # 创建多分类交叉熵损失函数对象
    loss_fn = nn.CrossEntropyLoss()
    # 计算损失
    loss = loss_fn(y_logits, y_true)
    print(f"多分类交叉熵损失函数：{loss}")

if __name__ == "__main__":
    demo01()

