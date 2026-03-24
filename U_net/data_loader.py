"""
关于数据读取的方法
用于处理数据样本的代码可能会变得混乱且难以维护；为了更好的可读性和模块化，我们理想情况下希望将数据集代码与模型训练代码解耦。PyTorch 提供了两个数据原语：torch.utils.data.DataLoader 和 torch.utils.data.Dataset，它们允许您使用预加载的数据集以及您自己的数据。Dataset 存储样本及其对应的标签，而 DataLoader 将一个可迭代对象包装在 Dataset 周围，以便于访问样本。

PyTorch 域库提供了一些预加载的数据集（例如 FashionMNIST），它们继承自 torch.utils.data.Dataset 并实现了特定于该数据集的功能。它们可用于原型设计和模型基准测试。您可以在此处找到它们：图像数据集、文本数据集 和 音频数据集

加载数据集
"""
import torch
from torch.utils.data import dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt 

"""
我们使用以下参数加载 FashionMNIST 数据集
root 是存储训练/测试数据的路径，

train 指定训练集或测试集，

download=True 如果数据在 root 目录下不可用，则从互联网下载数据。

transform 和 target_transform 指定特征和标签的转换"""
training_data = datasets.FashionMNIST(
    root = "data",
    train =True,
    download= True,
    transform=ToTensor()

)
testing_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download= True,
    transform=ToTensor()
)
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8,8))
cols,rows =3,3
for i in range(1,cols*rows+1):
    sample_idx =torch.randint(len(training_data),size=(1,)).item()
    img,label=training_data[sample_idx]
    figure.add_subplot(cols,rows,i)
    plt.title(labels_map[label])
    plt.axis('off')
    plt.imshow(img.squeeze(),cmap='grey')

plt.show()
