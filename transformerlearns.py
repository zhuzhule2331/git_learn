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
    

class FeedForward(nn.Module):
    """
    前馈神经网络 (Position-wise Feed Forward)

    公式: FNN (x)=Max(0,x * w1+b1)*w2 +b2

    特点：
    -对每个位置进行独立的相同的操作
    -包含两个线性变换和一个ReLU激活
    -通常中间层的维度是模型维度的4倍
    使用场景：
    -增加模型的非线性表达能力
    -在注意力机制后进一步表达特征
    """
    def __init__(self,d_model:int,d_ff:int =None,dropout:float =0.1):
        """
        参数说明：
        d_model:模型维度
        d_ff:前馈网络的中间层维度(默认为4*d_model)
        dropout:Dropout概率"""
        super(FeedForward,self).__init__()

        # 如果没有指定d_ff,默认为d_model的4倍（transformer原文中的设定）
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.d_model = d_model
        self.d_ff = d_ff
        # 两个线性层
        self.linear1 =nn.Linear(d_model,d_ff) #[d_model,d_ff]
        self.linear2 = nn.Linear(d_ff,d_model) #[d_ff,d_model]

        #Dropout 层
        self.dropout =nn.Dropout(dropout)
        self.activation = nn.ReLU()
        print("😂✔️✔️前馈神经网络初始化完成")
        print(f"  模型维度{d_model}")
        print(f"  中间层维度{d_ff}")

        print()

    def forward(self,x:torch.Tensor)->torch.Tensor:
        """
        前向传播
        输入:
        [batch_size,seq_len,d_model]
        输出：
        [batch_size,seq_len,d_model]
        
        数据流示例：
            输入：[32,100,512] #32个样本，100个位置，512维
            步骤1：第一个线性层
            [32,100,512]->[32,100,2048] #扩展到4倍
            步骤2：ReLU激活
            [32,100,2048]->[32,100,2048]
            步骤3：Dropout
            [32,100,2048]->[32,100,2048]
            步骤4：第二个线性层
            [32,100,2048]->[32,100,512] #压缩回原始维度
            步骤5：Dropout
            [32,100,512]->[32,100,512]"""
        # 第一个线性变换+激活函数
        hidden=self.linear1(x) #[batch_size,seq_len,d_model]->[batch_size,seq_len,d_ff]
        hidden=self.activation(hidden) #ReLU激活
        hidden=self.dropout(hidden)  #Dropout
        
        # 第二个线性变换
        output=self.linear2(hidden) # [batch_size,seq_len,d_ff]->[batch_size,seq_len,d_model]
        output=self.dropout(output) # Dropout


        return output
      

class LayerNorm(nn.Module):
    """
    层归一化：
    为什么 LayerNorm 适合序列任务？
    BatchNorm：对批次维度归一化（比如[batch, seq, dim]，对 batch 维度求均值），但序列任务中 seq_len 可变，batch 内样本分布不稳定；
    LayerNorm：对每个样本的特征维度归一化（比如每个[seq, dim]的样本，对 dim 维度求均值），不依赖批次，更稳定。
    -BatchNorm在序列任务中效果不好（系列长度可变）
    -LayerNorm对每个样本独立进行归一化
    -有助于稳定训练加速收敛

    公式：y = γ *（x -μ）/σ +β

     μ：均值
     σ：方差
     γ：可学习的缩放参数
     β：可学习的偏移参数
    """
    def __init__(self,d_model:int,eps:float=1e-6):
        """
        参数说明：
        d_model:特征维度
        eps:防止除0的小数
        """
        super(LayerNorm,self).__init__()
        self.d_model =d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))# 初始化γ参数为1
        self.beta = nn.Parameter(torch.zeros(d_model))# 初始化β参数为0
        
        print(f"LayerNorm初始化完成！维度为{d_model}")


    def forward(self,x : torch.Tensor) -> torch.Tensor:
        """前向传播
        输入：x[batch_size,seq_len,d_model]
        输出：[batch_size,seq_len,d_model]
        数据流示例：

        输入：[32，100，512]

        步骤1：计算均值和方差（在最后一个维度上）
        mean:[32,100,1]
        var:[32,100,1]
        步骤2：归一化
        x_norm =(x-mean)/sqrt(var+eps)
        步骤3：缩放和偏移
        output = γ * x_norm + β
        

        
        """


        # 计算均值和方差（在最后一个维度上）
        mean = x.mean(dim=-1,keepdim=True)#[batch_size,seq_len,1]
        var = x.var(dim=-1,keepdim=True)# [batch_size,seq_len,1]

        #归一化
        x_norm = (x-mean)/torch.sqrt(var+self.eps)
        #缩放和偏移
        output = x_norm * self.gamma +self.beta

        return output
        

class EncoderLayer(nn.Module):
    """
    Transformer编码器层
    结构：
    1.多头注意力机制
    2.Add & Norm (残差链接+层归一化)
    3.前馈网络
    4.Add & Norm (残差链接+层归一化)
    使用场景：
    -文本编码：BERT，GPT的编码器部分
    -图像编码：VIT(vision Transformer)
    -音频编码：音频特征提取

    """
    def __init__(self,d_model:int,n_heads:int,d_ff:int = None,dropout:float =0.1):
        """参数说明：
           d_model:模型维度
           n_heads:注意力头数
           d_ff:前馈网络中间层维度
           dropout:Dropout概率
        
        """
        super(EncoderLayer,self).__init__()

        # 多头自注意力机制
        self.self_attention= MultiHeadAttention(d_model,n_heads,dropout)

        #前馈网络
        self.feed_forward = FeedForward(d_model,d_ff,dropout)

        # 两个LayerNorm层
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        print(f'✔️编码器初始化完成！')

    def forward(self,x:torch.Tensor,mask:Optional[torch.Tensor] = None)->torch.Tensor:
        """前向传播
            输入：
            x:[batch_size,seq_len,d_model] -输入序列
            mask:[batch_size,1,1,seq_len] -注意力掩码（可选）
            输出：
            [batch_size,seq_len,d_model]-编码后的序列
            数据流示例（文本编码）：
            输入句子：“我爱北京天安门”（七个字）
            x:[32，7，512]#32个样本，7个字，512维
            步骤1：自注意力
            attn_output :[32,7,512]
            步骤2：残差连接+LayerNorm
            x=Layernorm(x+attn_output)
            步骤3：前馈网络
            ff_output：[32,7,512]
            步骤4：残差连接 +LayerNorm
            x = LayerNorm(x+ff_output)

            输出:[32,7,512]


        
        
        """
        # 子层1：自注意力
        # 保存残差
        residual = x
        # 自注意力（Q=K=V都是x）
        # 因为编码器使用自注意力，Q/K/V 必须来自同一个输入序列 x，目的是让序列内部各位置互相关注。
        attn_output,_ = self.self_attention(x,x,x,mask)
        #Dropout
        attn_output = self.dropout(attn_output)
        #残差连接+LayerNorm
        x = self.norm1(attn_output+residual)
        # 子层2：前馈神经网络
        # 保存残差
        residual = x
        # 前馈网络
        ff_output = self.feed_forward(x)
        # 残差连接+LayerNorm
        x = self.norm2(ff_output+residual)
        return x
    

class DecoderLayer(nn.Module):
    """Transformer解码器层：
       结构：
       1.mask多头自注意力机制（防止看到未来信息）
       2.Add & Norm
       3.编码器解码器交叉注意力机制
       4.Add & Norm
       5.前馈网络
       6.Add & Norm

       使用场景：
       -机器翻译：将源语言翻译为目标语言
       -文本生成：GPT系列
       -图像描述：根据图像生成对应描述    
    """
    def __init__(self,d_model:int,n_heads:int,d_ff:int,dropout:float = 0.1):
        """
        参数说明
        d_model:模型维度
        n_head:注意力头数
        d_ff:前馈网络中间层维度
        dropout：Dropout概率"""

        super(DecoderLayer,self).__init__()
        # Mask自注意力（用于目标序列）
        self.mask_self_attention = MultiHeadAttention(d_model,n_heads,dropout)
        # 交叉注意力（用于关注编码器输出）
        self.cross_attention = MultiHeadAttention(d_model,n_heads,dropout)
        #前馈网络
        self.feed_forward = FeedForward(d_model,d_ff,dropout)
        #定义三个Layernormal层
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        # Dropout层
        self.dropout =nn.Dropout(dropout)

        print(f"解码器层初始化完成")

    def forward(self,x:torch.Tensor,encoder_output:torch.Tensor,src_mask:Optional[torch.Tensor]=None,
                tgt_mask:Optional[torch.Tensor]=None)->torch.Tensor:
        """
        前向传播：
        输入：
         x[batch_size,tgt_len,d_model] - 目标序列
         encoder_output：[batch_size,src_len,d_model] - 编码器输出
         src_mask:[batch_size,1,1,src_len]- 源序列掩码
         tgt_mask:[batch_size,1,tgt_len,tgt_len]- 目标序列掩码
        
        输出：
          [batch_size,tgt_len,d_model]-解码后的序列
        数据流示例（中英翻译）：
         源句子：“我爱北京”（编码器已处理）
         目标翻译：“I Love Beijing”（正在生成）
         x ：[32,3,512] # “I Love Beijing
         encoder_output:[32,4,512] # "我爱北京“的编码
        步骤1：masked的自注意力（只能看到已经生成的词）
            生成Beijing时，只能看到I love
        步骤2：交叉注意力（关注原句子）
            决定“Beijing”应对应“北京”
        步骤3：前馈神经网络
            进一步处理特征

        
        """
        # 子层1：masked自注意力
        residual =x 
        # masked的自注意力
        mask_attn_output , _ = self.mask_self_attention(x,x,x,tgt_mask)
        mask_attn_output = self.dropout(mask_attn_output)

        # 参数连接 +LayerNorm
        x = self.norm1(residual + mask_attn_output)

        # 子层2：交叉自注意力机制
        residual = x
        # 交叉子注意力机制，Q来自解码器，K，V来自编码器
        cross_attn_output,_ = self.cross_attention(x#Q来自解码器
                                                 ,encoder_output#K来自编码器
                                                 ,encoder_output#V来自编码器
                                                 ,src_mask)
        cross_attn_output = self.dropout(cross_attn_output)
        # 残差连接+LayerNorm
        x = self.norm2(x+cross_attn_output)

        # 子层3： 前馈神经网络
        residual = x

        ff_output = self.feed_forward(x)
        #ff_output = self.dropout(ff_output)
        # 残差连接+LayerNorm
        x = self.norm3(residual+ff_output)
        return x
    
class TransformerEncoder(nn.Module):
    """完整的Transformer编码器:
       包含   1.词嵌入层
              2.位置编码
              3.N个编码器层的堆叠

        使用场景：
              1.BERT：双向语言理解
              2.文本分类：将文本编码为特征向量
              3.特征提取：提取图片/文本的高级特征

    """
    def __init__(self, vocab_size:int,
                 d_model:int=512
                 ,n_heads:int=8,n_layers:int = 6
                 ,d_ff:int=2048,
                 max_len:int=5000,
                 dropout:float = 0.1):
        super().__init__()
        """
        vocab_size：词汇表大小
        d_model:模型维度
        n_heads:注意力头数
        n_layers:编码器层数
        d_ff:前馈神经网络维度
        max_len:模型最大维度
        dropout:Dropout概率

        """    
        self.d_model =d_model
        self.n_layers = n_layers
        # 词嵌入层（将词ID转化为向量）
        self.embedding = nn.Embedding(vocab_size,d_model)
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model,max_len)
        #堆叠N个编码器层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model,n_heads,d_ff,dropout) for _ in range(n_layers)
        ])
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        # 初始化词嵌入层权重
        self._init_embeddings()

        print(f"✔️Transformer编码器初始化完成！")
        print(f"  词汇表大小{vocab_size}")
        print(f"  编码器层数{n_layers}")
        print(f"  模型维度{d_model}")

    def _init_embeddings(self):
        """初始化词嵌入层权重"""
        # 使用正态分布初始化
        nn.init.normal_(self.embedding.weight,mean=0,std=self.d_model**-0.5)
    
    def forward(self,
               src:torch.Tensor,
               src_mask:Optional[torch.Tensor] = None)->torch.Tensor:
        """
               前向传播
               输入：
               src:[batch_size,src_len]-源序列的词Id
               src_mask [batch_size,1,1,src_len]-源序列掩码
               输出：
               [batch_size,seq_len,d_model]-编码后的序列
               数据流示例（文本编码）：
               输入句子Id：[101,2023,456,102]#4个词的ID
               src:[32,4]# 32个样本，4个词
               步骤1：词嵌入
               [32,4] -> [32,4,512]
               步骤2：缩放（为了和位置编码平衡）
                [32,4,512] *sqrt(512)
               步骤3：位置编码
                [32,4,512] + 位置编码
               步骤4：通过6个编码器层
                [32,4,512] ->[32,4,512]
                [32,4,512] ->[32,4,512]
                 ...
                [32,4,512] ->[32,4,512]
                [32,4,512] ->[32,4,512]
                输出：[32,4,512]
               


        """
         #获取序列长度和批次大小
        batch_size,seq_len =src.shape
        # 步骤1：词嵌入
        x = self.embedding(src)#[batch_size,seq_len,d_model]
        # 步骤2：缩放嵌入（transformer中的技巧）
        x = x*math.sqrt(self.d_model)
        # 步骤3：添加位置编码
        x = self.positional_encoding(x)
        #Dropout
        x = self.dropout(x)

        # 步骤4：通过N个编码器层
        for i,layer in enumerate(self.layers):
            x = layer(x,src_mask)
            print(f"第{i+1}层编码器输出shape{x.shape}")

        
        return x

class TransformerDecoder(nn.Module):
    """
    完整的transformer解码器
      包含：
      -词嵌入层
      -位置编码
      -N个解码器层的堆叠
      -输出层（词汇表概率）
      使用场景
      机器翻译：生成目标语言
      文本生成：GPT系列
      序列到序列任务：摘要生成，对话系统
    """
    def __init__(self,
                 vocab_size:int,
                 d_model:int= 512,
                 n_heads:int = 8,
                 n_layers:int=6,
                 d_ff:int = 2048,
                 max_len:int = 5000,
                 dropout:float = 0.1):
        """
        参数说明：
        vocab_size:词汇表大小
        d_model:模型维度
        n_heads:注意力头数
        n_layers：解码器层层数
        d_ff：前馈网络维度
        max_len:最大序列长度
        dropout：Dropout概率
        """
        super(TransformerDecoder,self).__init__()
        self.d_model =d_model
        self.n_heads =n_heads
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size,d_model)
        # 位置编码
        self.positionalenconding = PositionalEncoding(d_model,max_len)
        # 添加N个解码器层
        self.layers  = nn.ModuleList([
            DecoderLayer(d_model,n_heads,d_ff,dropout) for _ in range(n_layers)
        ])
        
        # 输出层，将d_modle维映射到vocab_size（预测下一个词）
        self.output_projection = nn.Linear(d_model,vocab_size)
    
        #Dropout
        self.dropout = nn.Dropout(dropout)

        #初始化权重
        self._init_weights()

         
        print(f"✔️解码器初始化完成")
        print(f"  词汇表大小{vocab_size}")
        print(f"  解码器层数{n_layers}")
        print(f"  模型维度{d_model}")


    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.embedding.weight,mean= 0 ,std= self.d_model **-0.5 )
        nn.init.xavier_uniform_(self.output_projection.weight)
    
    def forward(self,
                tgt:torch.Tensor,
                encoder_output:torch.Tensor,
                src_mask:Optional[torch.Tensor] = None,
                tgt_mask:Optional[torch.Tensor] = None
                )->torch.Tensor:
        """ 前向传播
        输入：
        tgt:[batch_size,tgt_len] -目标序列的词ID
        encoder_output:[batch_size,src_len,d_model] -编码器输出
        src_mask:[batch_size,1,1,src_len] - 源序列掩码
        tgt_mask:[batch_size,1,tgt_len,tgt_len] - 目标序列掩码

        输出：
        [batch_size,tgt_len,vocab_size] -每个位置的词汇表概率
        数据流示例：文本翻译任务
        原序列：“我爱北京”（已编码）
        目标序列：“I Love Beijing”
        tgt :[32,3] #32个样本，3个词
        encoder_output:[32,4,512] # 编码器输出
        步骤1：词嵌入+位置编码
         [32,3] ->[32,3,512]
        步骤2：通过六个解码器层
          每个解码器层都会关注编码器输出
        步骤3：输出投影
          [32,3,512]->[32,3,vocab_size]
          得到每个位置的词汇表概率

        """
        # 获取目标序列的长度
        batch_size,tgt_len = tgt.shape
        # 步骤1：词嵌入
        x = self.embedding(tgt) #[batch_size,tgt_len,d_model]
        # 步骤2：缩放嵌入
        x = x * math.sqrt(self.d_model)
        # 步骤3：位置编码
        x = self.positionalenconding(x) #[batch_size,tgt_len,d_model]
        # Dropout 
        x = self.dropout(x)
        # 步骤4：通过N个解码器层
        for i,layer in enumerate(self.layers):
            x = layer(x,encoder_output,src_mask,tgt_mask)
            print(f"第{i}词解码后的形状为{x.shape}")
        # 步骤5：输出投影（预测词汇表概率）
        output = self.output_projection(x)# [batch_size,tgt_len,vocab_size]

        return output



# 测试前馈网络
def test_feedforward():
    """测试前馈神经网络"""
    print("\n" + "="*50)
    print("🧪 测试前馈网络")
    print("="*50)
    
    batch_size = 2
    seq_len = 10
    d_model = 512
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入 shape: {x.shape}")
    
    # 创建前馈网络
    ff = FeedForward(d_model)
    
    # 前向传播
    output = ff(x)
    print(f"输出 shape: {output.shape}")
    print(f"✅ 前馈网络测试通过！\n")
    
    return output

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


# 测试完整编码器
def test_encoder():
    """测试Transformer编码器"""
    print("\n" + "="*50)
    print("🧪 测试Transformer编码器")
    print("="*50)
    
    # 参数设置
    vocab_size = 10000
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    n_layers = 6
    
    # 创建编码器
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers
    )
    
    # 创建输入（随机的词ID）
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"输入shape: {src.shape}")
    
    # 前向传播
    output = encoder(src)
    print(f"输出shape: {output.shape}")
    print(f"✅ 编码器测试通过！\n")
    
    return output

# 测试完整解码器
def test_transformer_decoder():
    """测试TransformerDecoder的完整性和输出维度"""
    # 1. 配置测试参数（简化版，方便验证）
    vocab_size = 1000  # 词汇表大小
    d_model = 128      # 模型维度（缩小，加快测试）
    n_heads = 4        # 注意力头数（128/4=32，符合要求）
    n_layers = 2       # 解码器层数（简化）
    d_ff = 512         # 前馈网络维度
    batch_size = 2     # 批次大小
    src_len = 5        # 源序列长度（比如输入5个词）
    tgt_len = 3        # 目标序列长度（比如输出3个词）
    
    # 2. 初始化解码器
    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff
    )
    
    # 3. 构造测试输入
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))  # 目标序列词ID [2, 3]
    encoder_output = torch.randn(batch_size, src_len, d_model)  # 编码器输出 [2, 5, 128]
    src_mask = None  # 简化测试，暂不使用掩码
    tgt_mask = None  # 简化测试，暂不使用掩码
    
    # 4. 前向传播
    with torch.no_grad():  # 禁用梯度，加快测试
        output = decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    # 5. 验证输出维度
    print("\n===== 测试结果 =====")
    print(f"输入目标序列形状: {tgt.shape}")
    print(f"编码器输出形状: {encoder_output.shape}")
    print(f"解码器输出形状: {output.shape}")
    
    # 断言验证（核心维度检查）
    assert output.shape == (batch_size, tgt_len, vocab_size), \
        f"输出维度错误！预期 {(batch_size, tgt_len, vocab_size)}，实际 {output.shape}"
    print("✅ 维度验证通过！")
    
    # 6. 额外验证：权重初始化和前向传播无报错
    print("✅ 解码器前向传播无报错！")
    print("✅ 测试全部通过！")

class Transformer(nn.Module):
    """
    完整的transformer模型
    结合了解码器和编码器，实现了序列到序列的转换
    使用场景：
    -机械翻译：将一种语言转换为另一种语言
    -文本摘要：将长文本压缩成短摘要
    -对话系统：根据上下文形成回复
    -代码生成：根据代码生成描述"""
    def __init__(self, 
                 src_vocab_size:int,
                 tgt_vocab_size:int,
                 d_model:int = 512,
                 n_heads:int = 8,
                 n_encoder_layers:int = 6,
                 n_decoder_layers:int = 6,
                 d_ff:int = 2048,
                 max_len:int = 5000,
                 dropout:float = 0.1
                 ):
        """
        参数说明：
        -src_vocab_size:源语言词汇表大小
        -tgt_vocab_size:目标语言词汇表大小
        -d_model:模型维度
        -n_heads:注意力头数
        -n_encoder_layers:编码器层数
        -n_encoder_layers:解码器层数
        -d_ff:前馈神经网络维度
        -max_len:最大序列长度
        -dropout:Dropout概率
        """
        super().__init__()
        #编码器
        self.encoder = TransformerEncoder(src_vocab_size,d_model,n_heads,n_encoder_layers,d_ff,max_len,dropout)
        #解码器
        self.decoder =TransformerDecoder(tgt_vocab_size,d_model,n_heads,n_decoder_layers,d_ff,max_len,dropout)

        print(f"Transformer初始化完成")
        print(f"  模型维度：{d_model}")
        print(f"  源语言词汇表大小{src_vocab_size}")
        print(f"  目标语言词汇表大小{tgt_vocab_size}")
        print(f"  解码器层数{n_encoder_layers}")
        print(f"  编码器层数{n_decoder_layers}")
    pass

if __name__ == '__main__':
    # _ = test_positional_encoding()
    # # 先激活环境，再运行python

    # # 运行测试
    # _ = test_attention()
    # # 运行测试
    # _ = test_multihead_attention() 
    # # 运行测试
    # _ = test_feedforward()
    # # 运行测试
    # _ = test_encoder()
    _ = test_transformer_decoder()
