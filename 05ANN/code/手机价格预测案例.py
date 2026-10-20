"""
技术选项：
    单标签多分类任务

神经网络训练工作流：
    1.准备数据
        训练集，验证集，测试集

    2.搭建神经网络

    3.模型训练
        1）向前传播
        2）计算损失
        3）梯度清零
        4）反向传播
        5）权重参数更新

    4.模型测试

"""
import os
import torch
from torch.utils.data import TensorDataset,DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from torchsummary import summary


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

device=torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f'当前设备:{device}')

torch.manual_seed(141)

# 1.准备数据集
def create_dataset():
    # 使用 __file__ 构建路径，不受运行目录影响
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, '..', 'data', '手机价格预测.csv')
    data=pd.read_csv(csv_path)

    # 划分数据集：特征
    x=data.iloc[:,:-1]
    y=data.iloc[:,-1]
    x=x.astype(np.float32)

    # 划分数据集：训练集与测试集
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=28,stratify=y)

    #numpy=>torch
    x_train=torch.tensor(x_train.values).to(dtype=torch.float32)
    x_test=torch.tensor(x_test.values).to(dtype=torch.float32)
    y_train=torch.tensor(y_train.values).to(dtype=torch.long)
    y_test=torch.tensor(y_test.values).to(dtype=torch.long)

    # 张量=>数据集对象
    train_dataset=TensorDataset(x_train,y_train)
    test_dataset=TensorDataset(x_test,y_test)

    return train_dataset,test_dataset


# 搭建神经网络
class PhonePriceModel(nn.Module):
    def __init__(self,input_dim=20,output_dim=4):
        super().__init__()
        self.linear1=nn.Linear(input_dim,256)
    
        self.linear2=nn.Linear(256,128)
      
        self.linear3=nn.Linear(128,64)
      
        self.linear4=nn.Linear(64,output_dim)

    def forward(self,x):
        x=self.linear1(x)
        x=torch.relu(x)

        x=self.linear2(x)
        x=torch.relu(x)

        x=self.linear3(x)
        x=torch.relu(x)

        x=self.linear4(x)
        return x
    
    def fit(self,train_dataset,epochs=10):
        # 创建数据加载器
        # 参数1：数据集对象
        # 参数2：batch_size:每个批次的样本数，32
        # 参数3：shuffle：是否打乱数据，True打乱，False不打乱
        # 参数4：drop_last=False：是否删除最后一个不完整的批次，False保留，True删除
        train_loader=DataLoader(dataset=train_dataset,batch_size=32,shuffle=True,drop_last=False)

        # 将模型移动到对应设备
        self.to(device)
   
        # 构建损失函数和优化器
        loss_fn=nn.CrossEntropyLoss()
        optimizer=optim.AdamW(self.parameters(),lr=0.001,weight_decay=0.01)

        # 定义指标列表，记录每轮的训练损失和准确率，用于可视化
        train_loss_list=[]
        train_accs=[]

        # 循环训练
        for epoch in range(epochs):
            # 1.开启训练模式
            super(PhonePriceModel,self).train() # 切换为训练模式
            # 2.记录本轮的损失,记录本轮的预测正确样本数,记录样本总数
            train_loss=0.0
            correct_count=0
            total_samples=0

            # 3.遍历训练数据加载器，每次加载一个批次的数据,直到遍历完整个数据集
            for x_batch, y_batch in train_loader:
                # 将数据移动到对应的设备上
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                # 前向传播
                pred_logits=self(x_batch)
                # 计算损失
                loss=loss_fn(pred_logits,y_batch)
                # 梯度清零，防止梯度累积
                optimizer.zero_grad()
                # 反向传播：计算梯度
                loss.backward()
                # 权重参数更新
                optimizer.step()
                # 更新训练指标，训练损失，训练样本数，预测正确的样本数
                train_loss+=loss.item()*x_batch.shape[0]
                total_samples+=x_batch.shape[0]
                correct_count+=(pred_logits.argmax(dim=1)==y_batch).sum().item()

            # 4.计算本轮的平均损失
            avg_loss=train_loss/total_samples
            # 计算本轮的平均准确率
            acc=correct_count/total_samples
            
            # 5.记录本轮的平均损失和平均准确率
            train_loss_list.append(avg_loss)
            train_accs.append(acc)

            # 6.打印本轮的训练信息
            print(f'Epoch [{epoch+1}/{epochs}], avg_Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}')

        # 保存训练模型
        os.makedirs('./models',exist_ok=True)
        model_path=os.path.join('./models','phone_price_model.pth')
        torch.save(self.state_dict(),model_path)
        print(f'模型已保存到{model_path}')
        return train_loss_list,train_accs

    def plot_train_process(self,train_loss_list,train_accs):    
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.plot(train_loss_list)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train loss'])
     

        plt.subplot(1,2,2)
        plt.plot(train_accs)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['train accuracy'])
        plt.show()

    @torch.no_grad() #装饰器：不计算梯度，节省显存，加速推理
    def evaluate(self,test_dataset,input_dim=20,output_dim=4):
        # 创建数据加载器
        test_loader=DataLoader(dataset=test_dataset,batch_size=32,shuffle=False,drop_last=False)
        # 1.初始化一个模型
        model=PhonePriceModel(input_dim=input_dim,output_dim=output_dim).to(device)
        # 2.加载训练好的模型参数
        model.load_state_dict(torch.load('./models/phone_price_model.pth'))
        # 定义损失函数
        loss_fn=nn.CrossEntropyLoss()
        # 模型测试，设置评估模式model.eval()
        model.eval()
        # 定义变量 记录测试的总损失，总样本数，预测正确的样本数
        test_loss=0.0
        total_samples=0
        correct_count=0

        for x,y in test_loader:
            # 将数据移动到对应设备
            x,y=x.to(device),y.to(device)
            # 前向传播
            pred_logits=model(x)
            # 计算损失
            loss=loss_fn(pred_logits,y)
 
            # 更新测试指标，测试损失，测试样本数，预测正确的样本数
            test_loss+=loss.item()*x.shape[0]
            total_samples+=x.shape[0]
            correct_count+=(pred_logits.argmax(dim=1)==y).sum().item()

        # 计算本轮的平均损失
        avg_loss=test_loss/total_samples
        # 计算本轮的平均准确率
        acc=correct_count/total_samples
        # 打印测试结果
        print(f'Test Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}')
        return avg_loss,acc
        



   
     
    
if __name__=='__main__':
    train_dataset,test_dataset= create_dataset()
    # print(f'train_dataset: {len(train_dataset)}')
    # print(f'test_dataset: {len(test_dataset)}')
    model=PhonePriceModel()
    # train_loss_list,train_accs= model.fit(train_dataset)
    # model.plot_train_process(train_loss_list,train_accs)
    test_loss,test_acc=model.evaluate(test_dataset) 
    


