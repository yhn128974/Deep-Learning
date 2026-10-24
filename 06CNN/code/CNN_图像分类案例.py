"""
1.准备数据
    CIFAR10数据集 
    
2.搭建神经网络
    卷积层： nn.Conv2d()
        提取图像局部特征=>特征图，计算方式 N=（W-F+2P）//S+1
    激活函数 Relu: nn.ReLU()
    池化层： nn.MaxPool2d()

        降采样，不会改变通道数，只会改变长宽：最大池化，平均池化
    展平操作： nn.Flatten()
        将多维数据展平为一维数据
    全连接层： nn.Linear()
        
3.模型训练
    定义损失函数：nn.CrossEntropyLoss()
    定义优化器：torch.optim.Adam()
4.模型测试

优化思路：
    1.增加卷积层，激活层，池化层的深度
    2.增加全连接层的节点数
    3.调整学习率
    4.调整优化器
    5.调整激活函数
    6.
"""
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from torchsummary import summary
import os

# 定义样本的批次大小
BATCH_SIZE=8
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'img')
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','models')
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, 'cifar10_cnn.pth')
device = "cuda" if torch.cuda.is_available() else "cpu"


# 1.准备数据
def create_dataloader():
    train_dataset=CIFAR10(
        root=data_dir,
        train=True,
        transform=ToTensor(),
        download=True
    )
    test_dataset=CIFAR10(
        root=data_dir,
        train=False,
        transform=ToTensor(),
        download=True
    )
    return train_dataset,test_dataset




# 2.搭建神经网络

class ImageModel(nn.Module):
    #1.初始化父类
    def __init__(self):
        super().__init__()
        # 定义需要的神经网络
        # 卷积层1
        self.conv1=nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
        # 池化层1
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        # 卷积层2
        self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=3,stride=1,padding=0)
        # 池化层2
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2,padding=0)

        # 全连接层 
        # 输入576，输出120
        self.Linear1=nn.Linear(in_features=16*6*6,out_features=120)
        self.Linear2=nn.Linear(in_features=120,out_features=84)
        # 输出10
        self.output=nn.Linear(in_features=84,out_features=10)
        
    # 2. 写出前向传播过程
    def forward(self,x):
        # 第一部分
        x=self.conv1(x)
        x=torch.relu(x)
        x=self.pool1(x)
        # 第二分布 卷积，激活，池化
        x=self.pool2(torch.relu(self.conv2(x)))


        # 注意全连接层只能处理2维数据，所以要将数据拉平展开到2维上,每个批次有n个图片
        # 参数一：x.size(0)计算一图片的大小
        # 参数二:-1 自动匹配图片个数
        x=x.reshape(x.size(0),-1) #8行576列
        # print(f'x.shape{x.shape}')

        # 接入全连接层
        x=torch.relu(self.Linear1(x))
        x=torch.relu(self.Linear2(x))
        x=self.output(x)
        return x

    # 3.模型训练
    def train_model(self,train_dataset):
        # 1.数据加载器
        data_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)

        # 2.创建模型对象!
        model=ImageModel()

        # 3.定义损失函数! nn.CrossEntropyLoss()
        loss_fn=nn.CrossEntropyLoss()

        # 4.定义优化器! torch.optim.Adam()
        optimizer=optim.Adam(model.parameters(),lr=1e-3)

        # 5.定义循环参数，进行训练
        epochs=10
        for epoch in range(epochs):
            # 定义变量记录，单次总损失，样本总数，正确预测数量,训练开始时间
            total_loss=0
            correct_predictions=0
            total_samples=0
            start_time=time.time()
            # 遍历加载器获取每批次的数据的特征和标签
            for x,y in data_loader:
                # 进入训练模型
                model.train()
                # 模型预测
                y_pre=model(x)
                # 计算损失
                loss=loss_fn(y_pre,y)

                # 梯度清零，反向传播，参数更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # ：获取模型预测概率/得分最大的类别索引（即模型最终预测出来的类别标签）。
                # print(torch.argmax(y_pre,dim=1))
              
                # print(y) #真实标签
            
                # print(torch.argmax(y_pre,dim=1)==y) #预测值==真实值，返回True/False
       
                # print((torch.argmax(y_pre,dim=1)==y).sum()) #返回True/False的数量

                # 统计当前正确预测的个数
                correct_predictions+=(torch.argmax(y_pre,dim=1)==y).sum()
                # 总损失 ：累加=》本批平局损失*样本数
                total_loss+=loss.item()*len(y)
                # 总样本数
                total_samples+=len(y)

                # break
            # 打印该轮的训练信息
            print(f'第{epoch+1}轮训练，用时{time.time()-start_time:.2f}秒,平均损失{total_loss/total_samples:.4f},准确率{correct_predictions.item()/total_samples:.4f}')
            # break

        # 6.保存模型
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存至: {model_path}")

    # 4.模型测试
    def module_test(self,test_dataset):
        dataloader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)

        # 创建模型
        model=ImageModel()

        # 加载模型参数
        model.load_state_dict(torch.load(model_path))
        
        model.eval()
        
        # 定义变量统计正确预测的样本个数，总样本个数，总损失
        total_correct, total_samples = 0, 0

        # 遍历数据加载器，获取到每批次的数据
        for x,y in dataloader:
            model.eval()
            y_pre=model(x)
            # 这里用argmax来替代softmax+loss，可以更直观的计算准确率
            # argmax函数的功能；返回最大值对应的索引
            # 假设y_pre=[[0.1,0.2,0.7],[0.8,0.1,0.1]],dim=1表示在行上进行操作
            # 返回的是[2,0],这就是我们想要的预测值
            y_pre=torch.argmax(y_pre,dim=1)

            # 统计当前批次正确预测的样本个数
            total_correct += (y_pre == y).sum().item()
            # 统计当前批次的样本总数
            total_samples += len(y)

        # 打印正确率
        print(f'正确率{total_correct/total_samples:.4f}')



        


if __name__=='__main__':
    train_dataset,test_dataset=create_dataloader()
    # print(f'train_dataset shape:{train_dataset.data.shape}, test_dataset shape:{test_dataset.data.shape}')
    # print(f'train_dataset type:{train_dataset.class_to_idx}')

    # # 取出一个图像查看
    # image = train_dataset.data[0]
    # label = train_dataset.targets[0]
    # plt.imshow(image)
    # # {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    # plt.title(f"Label: {label}")
    # plt.axis('off')
    # plt.show()

       # 自动选择 GPU 或 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 实例化模型并迁移到设备上
    model = ImageModel().to(device)

    # # 查看模型结构（传入 device）
    # summary(model, input_size=(3, 32, 32), BATCH_SIZE=1, device=device)

    # 模型训练
    model.train_model(train_dataset)
    # 模型评估
    model.module_test(test_dataset)






    