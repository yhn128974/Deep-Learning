"""
循环网络工作原理：
    1.上一次的隐藏状态 +  本次的输入  ===>  本次的隐藏状态===>输出;
    2.本次的隐藏状态作为下一次的隐藏状态；本次的输出可以作为下一次的输入


"""
import torch
import torch.nn as nn

# 参数说明：
# input_size:输入特征的维度
# hidden_size:隐藏状态的维度
# num_layers:循环神经网络的层数
rnn=nn.RNN(input_size=128,hidden_size=256,num_layers=1,batch_first=True)

# 一次性处理 5 句话，每句话有 32 个词，每个词由 128 维的词向量表示
"""
# 1. 第一个数字 5：批次大小（batch_size，样本数）
#    含义：一次同时送进神经网络处理的句子数量。
# 2. 第二个数字 32：序列长度（seq_len / time_steps，时间步）
#    含义：每句话包含的词语数量。
# 3. 第三个数字 128：特征维度（input_size / embedding_dim）
#    含义：每一个词转化为向量后的长度。
"""

# size=(32,5,128)的含义：
# 32: batch_size(样本数)   # 5:序列长度（分词个数）(time_steps)  # 128:特征的维度
x=torch.randn(size=(32,5,128))

# size=(1,32,256)的含义：
# 1: num_layers(层数)   # 32:batch_size(样本数)  # 256:隐藏状态的维度
h0=torch.randn(size=(1,32,256))

# 条用RNN进行预测
output,h1=rnn(x,h0)
print(f'output.shape:{output.shape},h1.shape:{h1.shape}')




