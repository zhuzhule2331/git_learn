"""学习了解vision transformers的代码使用"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange,repeat
from einops.layers.torch import Rearrange

class pre_proces(nn.Module):
    def __init__(self,image_size,patch_size,patch_dim,dim):
        super().init()
        self.patch_size  = patch_size
        self.dim =dim
        self.patch_num = (image_size//patch_size)**2 #这个是针对正方形的，所以只有一个尺寸，如果是长方形就是长//长 * 宽//宽
        self.linear_embedding = nn.linear(patch_dim,dim)
        self.position_embedding = nn.Parameter(torch.randn(1,self.patch_num+1,self.dim)) # 使用广播
        self.CLS_token = nn.Parameter(torch.ramdn(1,1,self.dim)) #别忘了维度要和（B，L，C）对齐

    def forward(self,x):
        x = rearrange(x,'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',p1 =self.patch_size,p2=self.patch_size) #(B,L,C)
        x =self.linear_embedding(x)
        b,l,c =x.shape #获取维度
        CLS_token =repeat(self.CLS_token,'1 1 d -> b 1 d',b=b) # 位置编码复制B份
        x =torch.concat((CLS_token,x),di =1)
        x= x+self.position_embedding
        return x
        
        





