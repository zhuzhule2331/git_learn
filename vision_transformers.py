"""学习了解vision transformers的代码使用"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange,repeat
from einops.layers.torch import Rearrange

class pre_proces(nn.Module):
    def __init__(self,image_size,patch_size,patch_dim,dim):
        super().__init__()
        self.patch_size  = patch_size # patch 的大小
        self.dim =dim #transformer的维度，transformer具有输入输出不变性
        self.patch_num = (image_size//patch_size)**2 #patch的个数这个是针对正方形的，所以只有一个尺寸，如果是长方形就是长//长 * 宽//宽
        self.linear_embedding = nn.Linear(patch_dim,dim) #线性嵌入层
        self.position_embedding = nn.Parameter(torch.randn(1,self.patch_num+1,self.dim)) # 使用广播
        self.CLS_token = nn.Parameter(torch.randn(1,1,self.dim)) #别忘了维度要和（B，L，C）对齐

    def forward(self,x):
        x = rearrange(x,'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',p1 =self.patch_size,p2=self.patch_size) #(B,L,C)
        x =self.linear_embedding(x)#线性嵌入
        b,l,c =x.shape #获取维度
        CLS_token =repeat(self.CLS_token,'1 1 d -> b 1 d',b=b) # 位置编码复制B份
        x =torch.cat((CLS_token,x),dim =1)#级联CLSToken
        x= x+self.position_embedding #位置嵌入
        return x

class Multihead_self_attention(nn.Module):
    def  __init__(self,heads,head_dim,dim):
         super().__init__()
         self.head_dim = head_dim #每一个注意力头的维度
         self.heads = heads #注意力头的个数
         self.inner_dim = head_dim*heads #多头自注意力最后的输出维度
         self.scale =self.head_dim**-0.5
         self.to_qkv = nn.Linear(dim,self.inner_dim*3) #生成Q，K，V，每一个矩阵的维度由自注意力头的维度以及头的个数决定
         self.to_output = nn.Linear(self.inner_dim,dim)#输出层
         self.norm = nn.LayerNorm(dim) #归一化层
         self.softmax = nn.Softmax(dim =-1)

    def forward(self,x):
        x =self.norm(x) #pre -norm
        qkv =self.to_qkv(x).chunk(3,dim=-1) #划分Q，K，V，返回一个列表，其中就包含了QKV
        Q,K,V = map(lambda t:rearrange(t,'b l (h dim) -> b h l dim',dim =self.head_dim),qkv)
        K_T = K.transpose(-1,-2)
        att_score = Q@K_T*self.scale
        att =self.softmax(att_score)
        out =att@V
        out = rearrange(out,'b h l dim -> b l (h dim)')#拼接
        out =self.to_output(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self,dim,mlp_dim):
        super().__init__()
        self.fc1 =  nn.Linear(dim,mlp_dim)
        self.fc2 = nn.Linear(mlp_dim,dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self,x):
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Transformer_block(nn.Module):
    def __init__(self,dim,heads,head_dim,mlp_dim):
        super().__init__()
        self.MHA =Multihead_self_attention(heads=heads,head_dim=head_dim,dim=dim)
        self.FeedForward= FeedForward(dim =dim,mlp_dim=mlp_dim)

    def forward (self,x):
        x=self.MHA(x)+x
        x = self.FeedForward(x)+x
        return x
    
class ViT(nn.Module):
    def __init__(self,image_size,channels,patch_size,dim,heads,head_dim,mlp_dim,depth,num_class):
        super().__init__()
        self.to_patch_embedding = pre_proces(image_size = image_size,patch_size=patch_size,patch_dim = channels*patch_size**2,dim=dim)
        
        self.transformer=Transformer_block(dim=dim,heads=heads,head_dim=head_dim,mlp_dim=mlp_dim)
        self.MLP_head = nn.Sequential(nn.LayerNorm(dim),
                                      nn.Linear(dim,num_class))
        self.softmax=nn.Softmax(dim=-1)
    
    def forward(self,x):
        token = self.to_patch_embedding(x)
        output =self.transformer(token)
        CLS_token = output[:,0,:]#提取出CLS Token
        out =self.softmax(self.MLP_head(CLS_token))
        return out

    

        





