import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from vision_transformers import ViT
import os
# 设置PyTorch数据集国内镜像
os.environ['TORCH_DATASETS_URL'] = 'https://mirrors.tuna.tsinghua.edu.cn/pytorch/datasets/'

# ===================== 1. 设备配置（CPU训练） =====================
device = torch.device("cpu")

# ===================== 2. 超参数设置 =====================
IMAGE_SIZE = 32
CHANNELS = 3
PATCH_SIZE = 4
DIM = 128
HEADS = 4
HEAD_DIM = 32
MLP_DIM = 512
DEPTH = 3  # 堆叠3层Transformer Block
NUM_CLASS = 10
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4

# ===================== 3. 数据集（CIFAR10） =====================
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
val_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===================== 4. 初始化ViT模型（修复depth多层堆叠BUG） =====================
class ViT(nn.Module):
    def __init__(self,image_size,channels,patch_size,dim,heads,head_dim,mlp_dim,depth,num_class):
        super().__init__()
        patch_dim = channels * patch_size ** 2
        self.to_patch_embedding = pre_proces(image_size = image_size,patch_size=patch_size,patch_dim=patch_dim,dim=dim)
        
        # 堆叠depth层Transformer Block
        self.transformer_blocks = nn.ModuleList([
            Transformer_block(dim=dim,heads=heads,head_dim=head_dim,mlp_dim=mlp_dim)
            for _ in range(depth)
        ])
        self.MLP_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,num_class)
        )

    def forward(self,x):
        x = self.to_patch_embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        cls_token = x[:, 0, :]
        logits = self.MLP_head(cls_token)
        return logits

# 实例化模型
model = ViT(
    image_size=IMAGE_SIZE,
    channels=CHANNELS,
    patch_size=PATCH_SIZE,
    dim=DIM,
    heads=HEADS,
    head_dim=HEAD_DIM,
    mlp_dim=MLP_DIM,
    depth=DEPTH,
    num_class=NUM_CLASS
).to(device)

# ===================== 5. 损失函数+优化器 =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ===================== 6. 训练函数 =====================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(loader)
    acc = correct / total
    return avg_loss, acc

# ===================== 7. 验证函数 =====================
def val_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(loader)
    acc = correct / total
    return avg_loss, acc

# ===================== 8. 主训练循环 + 保存最优权重 =====================
best_acc = 0.0
save_path = "./vit_best.pth"

for epoch in range(1, EPOCHS+1):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = val_epoch(model, val_loader, criterion)

    print(f"【Epoch {epoch}/{EPOCHS}】")
    print(f"Train Loss:{train_loss:.4f} | Train Acc:{train_acc:.4f}")
    print(f"Val Loss:{val_loss:.4f} | Val Acc:{val_acc:.4f}\n")

    # 保存最优模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }, save_path)
        print(f"最优模型已保存至 {save_path}，最优精度:{best_acc:.4f}\n")

# ===================== 9. 加载权重测试 =====================
checkpoint = torch.load(save_path, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"加载模型成功，最优验证精度：{checkpoint['best_acc']:.4f}")