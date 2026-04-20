import torch
import torch.nn as nn
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet,self).__init__()
        self.linear1 = nn.Linear(3,10)
        self.relu =nn.ReLU()
        self.linear2 =nn.Linear(10,3)
        self.linear3=nn.Linear(3,1)

    def forward(self,x):
        residul = x # 残差连接,保留原始输入作为残差
        x=self.linear1(x)
        x =self.relu(x)
        x = self.linear2(x)+residul # 将残差添加到输出中
       
         # 通常后面会跟一个激活函数（可选）
        x = self.relu(x)
        output =self.linear3(x)
        return output
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
model = SimpleNet().to(device)


x = torch.arange(1,9999,0.5, dtype=torch.float32,device=device)

y = x * x + 2* x + x *x *x *x+x**0.5+x**5+6
y=y.unsqueeze(1) # 将y的形状从[19996]变为[19996, 1]
x=x.unsqueeze(1) # 将x的形状从[19996]变为[19996, 1]
print(x.shape)
print(y.shape)
model.train()
x_mean = x.mean(dim=0, keepdim=True)
x_std = x.std(dim=0, keepdim=True)
x = (x - x_mean) / (x_std + 1e-8)   # 特征标准化

y_mean = y.mean()
y_std = y.std()
y = (y - y_mean) / (y_std + 1e-8)           # 标签标准化
epochs = 100000
lr = 0.001
for epoch in range(epochs):
    pre = model(torch.cat([x**3, x*x, x], dim=1))
    loss = abs(pre - y).mean()
    for param in model.parameters():
        if param.grad is not None:
          param.grad.zero_()
    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param -= lr *param.grad
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        # 假设模型已定义并移动到 device
        print("最后一层权重形状:", model.linear3.weight.shape)
        print("最后一层权重数值:\n", model.linear3.weight)

        print("最后一层偏置形状:", model.linear3.bias.shape)
        print("最后一层偏置数值:\n", model.linear3.bias)

torch.save(model.state_dict(),"simple_net.pth")
print("模型已保存到 simple_net.pth")
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))
# import numpy as np
# a = torch.tensor([1,2.,3])
# print(a)
# b =torch.zeros(3)
# print(b)
# c = torch.ones([3,1])
# print(c)
# d = torch.rand([3,3])
# print(d)
# np_a = np.array([4,5.,6])
# print(np_a,np_a.dtype)
# torch_a = torch.from_numpy(np_a).float()
# print(torch_a, torch_a.dtype)
# x = torch.randn([3,5])
# print(x)
# print(x.reshape(5,3))
# b = x.view(5,3).clone()
# b=torch.ones([5,3])
# print(x)
# # print(x.flatten())
# # print(x[:,2])
# # print(x[0::2,0::])
# print(b)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# x = torch.randn([1,2],device=device,requires_grad=True)
# print(device)


# b = torch.randn([2,1],device=device,requires_grad=True)
# c = torch.randn([1],device=device,requires_grad=True)
# y = x @ b + c
# loss = y.sum()
# print(loss)
# loss.backward()
# print(x.grad)
# print(b.grad)
# print(c.grad)   
# import copy

# a = [[1, 2,[0]], [3, 4]]
# b = copy.copy(a[0])   # 浅拷贝

# b[2].append(99)
# b[1]=66
# print(a)           # [[1, 2, 99], [3, 4]]  原数据也被修改了
# print(b)           # [1, 2, 99]  新数据也被修改了
# d = copy.deepcopy(a)  # 深拷贝
# c = copy.copy(d)
# c.append(88)
# c[0].append(77)
# print(d)           # [[1, 2, 99], [3, 4]]  原数据没有被修改
