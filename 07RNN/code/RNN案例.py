"""
RNN歌词生成

"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import jieba
import os 

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path=os.path.join(script_dir,"../data/jaychou_lyrics.txt")

def build_vocab():
    # 1.定义一个变量记录：去重后所有的词，每行文本分词的结果
    unique_words,all_worlds=[],[]

    #2.遍历数据，将每行文本分词
    for line in open(data_path,'r',encoding='UTF-8'):
        words=jieba.lcut(line)
        # print(f'每行数据分词后的结果:{words}')
        all_worlds.append(words) #extend 把列表的内容添加进去
        #词语去重
        for word in words:
            if word not in unique_words:
                unique_words.append(word)

    # 统计不重复的词个数
    word_count=len(unique_words)
    # print(len(unique_words))

    # 3.构建词表，字典形式，key是词，value是词的索引
    word2idx={word:index for index,word in enumerate(unique_words)}
    # print(f'词表：{word2idx}')

    # 4.歌词文本用词的索引替代词本身，生成索引序列
    index_list=[]
    for words in all_worlds:
        temp=[]
        for word in words:
            temp.append(word2idx[word])
        
        # 再每行词之前添加空格隔开
        temp.append(word2idx[' '])

        # 获取文档中每个词的索引，添加到index_list中
        index_list.extend(temp)
        # print(f'每行数据的索引：{temp}')

    # 返回结果
    return index_list,unique_words,word2idx,word_count

    



    




    


if __name__=='__main__':
    # index_list：文档中每个词对应的索引
    # unique_words：不重复的词
    # word2idx：词表，字典形式，key是词，value是词的索引
    # word_count：不重复的词个数
   index_list,unique_words,word2idx,word_count= build_vocab()
#    print(f'文档中每个词对应的索引{index_list}')

