"""
演示
    梯度下降法的优化方法

介绍：
    梯度下降法，由损失函数的导数/梯度、学习率 来更新模型参数
    公式：w新 = w旧 - 学习率 * 梯度
    存在问题：
        1.平缓区域，梯度较小，参数更新慢
        2.“鞍点” ，梯度为 0，参数无法更新
        3.局部最小值，无法跳出，参数不是最优
梯度下降优化方法：
    1.调整梯度，比如使用指数移动加权平均来替代原始梯度
        动量法Momentum
            梯度的指数移动加权平均：st = β*st-1 + (1-β)*gt
            更新参数：wt = wt-1 - η*st
            
    2.调整学习率
        AdaGrad自适应学习率(Adaptive Gradient Estimation):
            累计平方梯度：
                St = St-1 + gt*gt
                其中，St为本次的累计平方梯度，St-1为上次的累计平方梯度，gt为本次的梯度
            学习率：
                η = η/(sqrt(St)+σ)
            更新参数公式：
                Wt = Wt-1 - η * 
            缺点：
                可能会使得学习率过早、过量的降低，导致模型训练后期学习率太小，较难找到最优解

        RMSProp(Root Mean Square Propagation)自适应学习率：
            对AdaGrad的优化，在计算累计平方梯度时，加入 调和权重系数
            指数加权累计平方梯度：
                St = β*St-1 + (1-β)*gt*gt
            其中St为本次的累计平方梯度，St-1为上次的累计平方梯度，gt为本次的梯度，β为 调和权重系数
            学习率：
                η = η/(sqrt(St)+σ)
            更新参数公式：
                Wt = Wt-1 - 调整后的η * gt
            优点：引入 调和权重系数，控制历史梯度 对 累计平方梯度的影响，防止累计平方梯度过大，从而防止学习率过早降低

        Adam(Adaptive Moment Estimation)自适应矩估计:
            同时优化 学习率 和 梯度
            公式：
                梯度(一阶矩)：
                    Mt = β1 * Mt-1 +(1-β1)*gt
                    Mt_hat = Mt / (1-β1^t)
                学习率(二阶矩):
                    St = β2 * St-1 + (1-β2)*gt*gt
                    St_hat = St / (1-β2^t)
                更新参数公式：
                    Wt = Wt-1 - 学习率/(sqrt(St_hat)+σ) * 调整后的梯度Mt_hat
            就是 RMSProp + Momentum


        AdamW:
            Adam的优化，解偶了权重衰减，在更新参数时直接添加了权重衰减项。
            公式:
                梯度(使用一阶矩)：
                    Mt = β1 * Mt-1 +(1-β1)*gt
                    Mt_hat = Mt / (1-β1^t)
                学习率(使用二阶矩):
                    St = β2 * St-1 + (1-β2)*gt*gt
                    St_hat = St / (1-β2^t)
                更新参数公式：
                    Wt = Wt-1 - 学习率/(sqrt(St_hat)+σ) * 调整后的梯度Mt_hat - η * λ * Wt-1
需要掌握：
    Adam,AdamW: optim.Adam/AdamW(实际使用)
    Momentum: optim.SGD(面试)

"""
import torch
import torch.nn as nn
import torch.optim as optim

# 动量发法
# 带动量的 SGD（即动量法 Momentum）
def momentum_method():
    # 初始化模型参数w
    w=torch.tensor([1.0],requires_grad=True,dtype=torch.float32)
    # 定义损失函数,w**2/2,导数=w
    loss =((w**2)/2)
    # 创建优化器(权重更新方法)，Momentum法
    optimizer=optim.SGD (params=[w],lr=0.01,momentum=0.9)
    # 计算梯度：梯度清零=》反向传播=》参数更新
    # 梯度清零，清空上一次的梯度值，避免对本次梯度有影响
    optimizer.zero_grad()
    # 反向传播得到梯度值，存放在w.grad中
    loss.backward()
    # 更新权重
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

    # 重复步骤第二次更新权重
    loss =((w**2)/2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')
  

# AdaGrad自适应学习率
def adagrad_method():
        # 初始化模型参数w
    w=torch.tensor([1.0],requires_grad=True,dtype=torch.float32)
    # 定义损失函数,w**2/2,导数=w
    loss =((w**2)/2)
    # 创建优化器(权重更新方法)，Adagrad法
    optimizer=optim.Adagrad (params=[w],lr=0.01)
    # 计算梯度：梯度清零=》反向传播=》参数更新
    # 梯度清零，清空上一次的梯度值，避免对本次梯度有影响
    optimizer.zero_grad()
    # 反向传播得到梯度值，存放在w.grad中
    loss.backward()
    # 更新权重
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

    # 重复步骤第二次更新权重
    loss =((w**2)/2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

# RMSProp自适应学习率
def rmsprop_method():
           # 初始化模型参数w
    w=torch.tensor([1.0],requires_grad=True,dtype=torch.float32)
    # 定义损失函数,w**2/2,导数=w
    loss =((w**2)/2)
    # 创建优化器(权重更新方法)，Adagrad法
    optimizer=optim.RMSprop (params=[w],lr=0.01,alpha=0.9)
    # 计算梯度：梯度清零=》反向传播=》参数更新
    # 梯度清零，清空上一次的梯度值，避免对本次梯度有影响
    optimizer.zero_grad()
    # 反向传播得到梯度值，存放在w.grad中
    loss.backward()
    # 更新权重
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

    # 重复步骤第二次更新权重
    loss =((w**2)/2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}') 

# Adam自适应矩估计
def adam_method():
   # 初始化模型参数w
    w=torch.tensor([1.0],requires_grad=True,dtype=torch.float32)
    # 定义损失函数,w**2/2,导数=w
    loss =((w**2)/2)
    # 创建优化器(权重更新方法,梯度下降方法)，Adagrad法,beta1为一阶矩的调和系数，beta2为二阶矩的调和系数
    optimizer=optim.Adam (params=[w],lr=0.01,betas=(0.9,0.99))
    # 计算梯度：梯度清零=>反向传播=>参数更新
    # 梯度清零，清空上一次的梯度值，避免对本次梯度有影响
    optimizer.zero_grad()
    # 反向传播得到梯度值，存放在w.grad中
    loss.backward()
    # 更新权重
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

    # 重复步骤第二次更新权重
    loss =((w**2)/2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}') 

# AdamW
def adamw_method():
    pass

if __name__ == '__main__':
    # momentum_method()
    # rmsprop_method()
    adam_method()