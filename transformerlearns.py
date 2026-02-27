import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional,Tuple

torch.manual_seed(42)
np.random.seed(42)#设置随机种子确保结果可复现

class PositionalEncoding(nn.Module):
    """位置编码模块:为序列中每个位置添加位置信息
    为什么需要位置编码？
    -transfomer不像RNN，没有循环结构，本身无法感知词的顺序
    -需要显示的告诉模型每个词在序列中的位置
    
    使用场景：
    1.文本：编码词在句子中的位置
    2.图像：编码patch在图像中的位置
    3.音频：编码帧在音频序列中的位置

    """
    def __init__(self,d_model:int,max_len:int=5000):
        """
        参数说明：
        d_model:模型的维度（例如512）
        max_len:支持的序列最大长度
        在 Transformer 中，输入的张量形状通常是 [batch_size, seq_len, d_model]，其中：
        batch_size：批次大小
        seq_len：序列长度（每个样本的 token 数量）
        d_model：每个 token 的特征维度（比如 512）
        """        
        super(PositionalEncoding,self).__init__()
        # 创建位置编码矩阵 shape[max_len,d_model]
        pe =torch.zeros(max_len,d_model)
        # 创建位置索引[0,1,2,...,max_len-1] shape[max_len,1]
        position = torch.arange(0,max_len).unsqueeze(1).float()

        # 计算分母项（用于计算正弦和余弦的周期）
        # 这里使用了一个技巧，用log和exp避免大数运算 shape[d_model/2]
        div_term = torch.exp(torch.arange(0,d_model,2).float()*
                             -(math.log(10000.0)/d_model))
        # 偶数维度使用sin shape[max_len.d_model/2]
        pe[:,0::2] = torch.sin(position*div_term)
        # 奇数维度使用cos shape[max_len,d_model/2]
        pe[:,1::2] = torch.cos(position*div_term)

        # 增加batch维度，并注册为buffer（不参与梯度更新）
        pe=pe.unsqueeze(0) #shape [1,max_len,d_model]
        self.register_buffer('pe',pe,False)
        #把一个张量（比如这里的 pe 位置编码矩阵）注册为模型的缓冲区（buffer） 
        # —— 它属于模型的一部分（会随模型移动到 GPU/CPU），但不会被优化器更新（非训练参数）
        print(f"✔️位置编码完成")
        print(f"最大序列长度对应于的seq_len单条序列中的样本数量{max_len}")
        print(f"模型维度{d_model}")

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        """
        前向传播
        输入：
        x[batch_size,seq_len,d_model]-输入的词的嵌入
        输出：
            [batch_size,seq_len,d_model]-添加位置掩码后的结果

        数据流示例：
        输入 x[32,100,512]#32个样本，每个100个词，每个词512维
        位置编码 [1,100,512]前100个位置的编码
        相加得到 [32，100，512] 添加位置编码后的结果

                    """
        seq_len = x.size(1)
        
        #获取对应长度的位置编码并相加
        #self.pe [:,:seq_len]会扩展到shape[1,seq_len,d_model]
        #广播机制会自动扩展到batch纬度
        output = x +self.pe[:,:seq_len]

        return output

def scaled_dot_product_attention(
        query:torch.Tensor,
        key:torch.Tensor,
        value:torch.Tensor,
        mask:Optional[torch.Tensor]=None,
        drop_out:Optional[nn.Dropout] = None
) -> Tuple[torch.Tensor,torch.Tensor]:
    """
    缩放点积注意力机制：
    公式：Attention(Q,K,V) =softmax(Q*k^T/sqrt(d_k))V
        K^T:K的转职
        d_k:K的维度
    为么要进行注意力缩放？
        -当d_k很大时，点积结果会很小
        -导致softmax后的梯度很小，训练困难
        -除以sqrt(d_k)可以解决这种问题
    参数：
        query:[batch_size,n_heads,seq_len,d_k] -查询矩阵
        key:[batch_size,n_heads,seq_len,d_k]-键矩阵
        value:[batch_size,n_heads,seq_len,d_v]- 值矩阵
        mask:[batch_size,1,1,seq_len] or [batch_size,1,seq_len,seq_len] -掩码
        drop_out:Drop_out层（可选）
            n_heads：注意力头的个数

    返回：
    output:[batch_size,n_heads,seq_len,d_v]-注意力输出
    attention_weights:[batch_size,n_heads,seq_len,seq_len]-注意力权重

    数据流示例：
        1.机器翻译 
        场景：我爱北京->I love Beijing.
        query:[32,8,10,64]#32个样本，8个头，10个词（token），每个词64维
        key:[32,8,10,64]
        value:[32,8,10,64]

        步骤1：Q*K^T ->[32,8,10,10]  # 每个词对每个词的注意力分数(k^T是K的转置)
            衡量每个 “查询（Q）” 和所有 “键（K）” 的匹配程度（分数越高，关联越强）
            比如文本中：第 i 个词对第 j 个词的注意力分数
        步骤2：缩放 ->[32,8,10,10]/sqrt(64) = [32,8,10,10]/8
        步骤3：softmax -> [32,8,10,10] # 归一化为概率 把注意力分数转化为 “权重”（概率），表示对每个位置的关注程度
        步骤4：乘以V->[32,8,10,64] # 加权求和得到输出  [32,8,10,10] × [32,8,10,64] → [32,8,10,64]	
                用注意力权重对 “值（V）” 加权，得到融合了全局关联信息的输出
        
    """
    # 获取最后一个维度的大小
    d_k = query.size(-1)

    #步骤1：计算Q*K^T
    # Q:[batch_size,n_heads，seq_len_q,d_k]
    # K的转置:[batch_size,n_heads,d_k,seq_len_k]
    # score:[batc_size,n_heads,seq_len_q,seq_len_k]
    scores = torch.matmul(query,key.transpose(-2,-1))
    #步骤2：缩放
    scores = scores/math.sqrt(d_k)
    #步骤3：如果有mask，应用mask（应用于masked Attention)
    if mask is not None:
        #mask为1的位置设为-inf,对应的softmax中的概率会变为0
        scores = scores.masked_fill(mask == 0,-1e9)
    #步骤4：Softmax归一化
    attention_weights = F.softmax(scores,dim=-1)
    #步骤5：如果有dropout 应用dropout
    if drop_out is not None:
        attention_weights =drop_out(attention_weights)
    #步骤6：乘以V得到输出
    # attentinon_weights:[batch_size,,n_heads,seq_len_q,seq_len_k]
    # value:[batch_size,h_heads,seq_len_k,d_v]
    # output:[batch_size,heads,seq_len_q,d_v]
    output = torch.matmul(attention_weights,value)
    return output,attention_weights

def test_positional_encoding():
    """测试位置编码模块"""
    print("\n" + "="*50)
    print("🧪 测试位置编码")
    print("="*50)
    
    batch_size = 2
    seq_len = 10
    d_model = 8
    
    # 创建模拟输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入shape: {x.shape}")
    
    # 创建位置编码层
    pos_encoder = PositionalEncoding(d_model, max_len=100)
    
    # 前向传播
    output = pos_encoder(x)
    print(f"输出shape: {output.shape}")
    print(f"✅ 位置编码测试通过！\n")
    
    return output

def test_attention():
    """测试注意力机制"""
    print("\n" + "="*50)
    print("🧪 测试缩放点积注意力")
    print("="*50)
    
    batch_size = 2
    n_heads = 4
    seq_len = 6
    d_k = 16
    
    # 创建Q, K, V
    Q = torch.randn(batch_size, n_heads, seq_len, d_k)
    K = torch.randn(batch_size, n_heads, seq_len, d_k)
    V = torch.randn(batch_size, n_heads, seq_len, d_k)
    
    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")
    
    # 计算注意力
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    print(f"输出 shape: {output.shape}")
    print(f"注意力权重 shape: {weights.shape}")
    print(weights)
    print(f"注意力权重和: {weights[0, 0, 0].sum():.4f} (应该接近1.0)")
    print(f"✅ 注意力机制测试通过！\n")
    
    return output, weights

def check_cuda_torch_info():
    torch_version = torch.__version__
    print(f"(●'◡'●)🔍torch的版本:{torch_version}")
    cuda_venv_version = torch.version.cuda
    print(f"🔎虚拟环境的cuda版本:{cuda_venv_version}")
    cuda_aviable = torch.cuda.is_available()
    print(f"cuda 可用👌" if cuda_aviable else "cuda 不可用😒")
    if cuda_aviable:
        gpu_count =torch.cuda.device_count()
        print(f"可用核心数：{gpu_count}")
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        print(f"当前Gpu索引{current_device},名称{gpu_name}")
        cuda_runtime_version = torch.backends.cudnn.version()
        print(f"cudnn的版本是{cuda_runtime_version}")

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    核心思想：
    -单个注意力可能只关注某一种关系
    -多个注意力可能关注不同的关系
    -最后将所有的头的输出拼接起来
    使用场景
    -文本：不同头关注语法，语义，长距离依赖等
    -图像：不同头关注纹理，颜色，形状等
    -多模态：不同头关注模态内和模态间的关系
    """
    def __init__(self,d_model:int,n_heads:int,dropout:float = 0.1):
        """
        参数
        d_model:模型维度（必须能被n_heads整除）
        n_head:注意力头数
        dropout:Dropout概率"""
        super(MultiHeadAttention,self).__init__()

        assert d_model%n_heads==0,f"d_model({d_model})必须要被n_heads({n_heads})整除"
        self.d_model=d_model
        self.n_heads=n_heads
        self.d_k=d_model//n_heads

        #创建Q，K，V的线性变换层（这里使用nn.Linear,不用高级API）
        #为什么是4个nn.Linear?
        #三个用于生成Q，K，V
        #剩下一个用于最后的映射输出
        self.w_q = nn.Linear(d_model,d_model,bias=False) #shape[d_model,d_model]
        self.w_k = nn.Linear(d_model,d_model,bias=False) #shape[d_model,d_model]
        self.w_v = nn.Linear(d_model,d_model,bias=False) #shape[d_model,d_model]
        self.w_o = nn.Linear(d_model,d_model,bias=False) #shape[d_model,d_model]

        self.dropout =nn.Dropout(dropout)
        #初始化权重
        self._init_weights()

        print("多头注意力机制初始化完成")
        print(f"    模型维度{d_model}")
        print(f"    注意力头数{n_heads}")
        print(f"    每个头的维度{self.d_k}")

    def _init_weights(self):
            """Xavier初始化权重"""
            for module in [self.w_q,self.w_k,self.w_v,self.w_o]:
                nn.init.xavier_uniform_(module.weight)
    
    def forward(self,
                query:torch.Tensor,
                key:torch.Tensor,
                value:torch.Tensor,
                mask:Optional[torch.tensor]=None
                )->Tuple[torch.Tensor,torch.Tensor]:
        """前向传播
        输入：
            query:[batch_size,seq_len_q,d_model]
            key:[batch_size,seq_len_k,d_model]
            value:[batch_size,seq_len_v,d_model]
            mask:[batch_size,1,1,seq_len] or None
            
        输出:
            output:[batch_size,seq_len_q,d_model]
            attention_weights[batch_size,seq_len_q,seq_len_k]
        
        数据流示例：
        文本翻译“Hello,World!(2个词)"
        Q，K，V[32,2,512] 32样本，2词，512维

        步骤1：线性变换（伪代码）
            Q = w_q*query -> [32,2,512]
            K = w_k*key -> [32,2,512]
            V = w_v*value ->[32,2,512]

        步骤2：分头（reshape+transpose
            Q->[32,2,512]->[32,2,8,64]->[32,8,2,64] #八个头，每个头64维
            K->[32,2,512]->[32,2,8,64]->[32,8,2,64]
            V->[32,2,512]->[32,2,8,64]->[32,8,2,64]
        步骤3：计算注意力
            每个头独立计算注意力  -> [32,8,2,64]
        步骤4：拼接所有头
            [32,8,2,64]->[32,2,8,64]->[32,2,512]
        步骤5：最终线性变换（伪代码，实际使用self.w_o(output)Linear类中方法）
            w_o * concat ->[32,2,512]
            
            """
        batch_size=query.size(0)
        seq_len_q =query.size(1)
        #步骤1线性变换生成Q，K，V
        #[batch_size,seq_len,d_model]->[batch_size,seq_len,d_model]
        Q = self.w_q(query) # 假设为[32,10,512]
        K = self.w_k(key)
        V = self.w_v(value)

        #步骤2：分头
        #[batch_size,seq_len,d_model]->[batch_size,seq_len,n_heads,d_k]->[batch_size,n_heads,seq_len,d_k]
        Q = Q.view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2) 
        K = K.view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        V = V.view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        # 现在shape变为[32,8,10,64]，-1是自动计算位数。

        #步骤3：计算缩放点积注意力
        #[batch_size,n_heads,seq_len_q,d_k]->[batch_size,n_heads,seq_len,d_k]
        attention_output,attention_weights=scaled_dot_product_attention(Q,K,V,mask,self.dropout)
        #output:[32,8,10,64]
        #attention_weights[32,8,10,10]

        #步骤4： 拼接多头输出
        #[batch_size,n_heads,seq_len_q,d_k]->[batch_size,seq_len_q,n_heads,d_k]
        # ->[batch_size,seq_len,d_model]
        attention_output=attention_output.transpose(1,2).contiguous().view(batch_size,seq_len_q,self.d_model)
        #现在[32,10,512]

        #步骤5：最终的线性变换
        output = self.w_o(attention_output)
        #output[32,10,512]
        return output,attention_weights
    
# 测试多头注意力
def test_multihead_attention():
    """测试多头注意力机制"""
    print("\n" + "="*50)
    print("🧪 测试多头注意力")
    print("="*50)
    
    batch_size = 2
    seq_len = 5
    d_model = 256
    n_heads = 8
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入 shape: {x.shape}")
    
    # 创建多头注意力层
    mha = MultiHeadAttention(d_model, n_heads)
    
    # 自注意力：Q=K=V
    output, weights = mha(x, x, x)
    
    print(f"输出 shape: {output.shape}")
    print(f"注意力权重 shape: {weights.shape}")
    print(f"✅ 多头注意力测试通过！\n")
    
    return output, weights



if __name__ == '__main__':
    _ = test_positional_encoding()
    # 先激活环境，再运行python

    # 运行测试
    _ = test_attention()
    # 运行测试
    _ = test_multihead_attention() 
