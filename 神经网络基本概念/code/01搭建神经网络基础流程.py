import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F

# 1. 数据准备
# 2.搭建神经网络
    # 1.定义类继承nn.modle
    # 2.__ini__初始化神经网络
    # 3.forword（）方法中完成向前传播
# 4.模型训练
# 3.模型测试

# TODO：搭建神经网络
class ModelDemo(nn.Module):
    # TODO:初始化
    def __init__(self):
        super().__init__()
        #  搭建创建隐藏层+输出层
        # 初始化隐藏层
        self.linear1=nn.Linear(3,3)
        self.linear2=nn.Linear(3,2)
        # 初始化输出层
        self.output=nn.Linear(2,2)

        #隐藏层权重参数初始化
        nn.init.xavier_uniform_(self.linear1.weight) 
        nn.init.zeros_(self.linear1.bias) 

        nn.init.kaiming_normal_(self.linear2.weight) 
        nn.init.zeros_(self.linear2.bias) 

    # TODO:前向传播
    def forward(self,x):
        # 得到激活值
        # z=self.linear1(x)
        # x=torch.sigmoid(z)
        # 指定激活函数 sigmoid,实现第一层计算
        x=torch.sigmoid(self.linear1(x))
        # 指定激活函数 relu,实现第二次计算
        x=torch.relu(self.linear2(x))
        # 指定激活函数 softmax,实现输出层计算,dim=1表示沿着行展开计算
        x=torch.softmax(self.output(x),dim=1)
        
        return x

# TODO:模型训练
def train():
    # 创建模型对象,自动调用forwold方法进行前向传播
    model=ModelDemo()#

    # 创建数据集样本，随机生成
    data=torch.randn(5,3)
    print(f"data:{data.shape},\n data:{data},\n data.requires_grad:{data.requires_grad}")

    # 调用神经网络进行模型运算
    output=model(data)
    print(f"output:{output}\n output_shape:{output.shape} \n output.requires_grad:{output.requires_grad}")

    # 打印计算和查看模型参数
    print('='*30)

    
    # 参数1：模型对象，参数2：输入数据形状
    summary(model,input_size=(5,3),device="cpu")

    # 打印模型各层结构
    print('='*30)
    for name,param in model.named_modules():
        print(f"参数名称：{name}，参数值：{param}")

    
    

    
    



    
# TODO:模型测试


if __name__=="__main__":
    train()


