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
        #self.pe [:,:seq_len]的shape[1,seq_len,d_model]
        #广播机制会自动扩展到batch纬度
        output = x +self.pe[:,:seq_len]

        return output

    def scaled_dot_product_attition(
            quary:torch.Tensor,
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
            quary:[batch_size,n_heads,seq_len,d_k] -查询矩阵
            key:[batch_size,n_heads,seq_len,d_k]-键矩阵
            value:[batch_size,n_heads,seq_len,d_k]- 值矩阵
            mask:[batch_size,1,1,seq_len] or [batch_size,1,seq_len,seq_len] -掩码
            drop_out:Drop_out层（可选）
        返回：

        数据流示例：
            
        """



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



if __name__ == '__main__':
    _ = test_positional_encoding()
    # 先激活环境，再运行python

