
"""
序列数据：后面的数据对前面的数据有依赖
RNN构成:

    词嵌入层：
        将词语转换成向量

    循环神经网络层：

    输出层：


"""
import torch
import jieba
import torch.nn as nn


def demo01():
    text = "我爱北京天安门"
    words = jieba.lcut(text)
    print("分词结果", words)

    #构建词嵌入层
    # 1.词汇表大小
    # 2.嵌入向量的维度,初始化分词函数
    embed=nn.Embedding(len(words),embedding_dim=4)

    print(f' embed:{embed}')

    # enumerate():返回索引和值
    # 第一个是索引，第二个是值
    for i,word in enumerate(words):
        # print(i,word)
        # 吧词索引转化为词向量

        world_vector=embed(torch.tensor(i)) #随机的，每次都不一样，无所谓
        print(f'{word}:{world_vector}')


if __name__=='__main__':
    demo01()