import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional,Tuple

torch.manual_seed(42)
np.random.seed(42)#设置随机种子确保结果可复现

class PoizitionalEnconding(nn.Module):
    """位置编码模块:为序列中每个位置添加位置信息
    为什么需要位置编码？
    -transfomer不像RNN，没有循环结构，本身无法感知词的顺序
    -需要显示的告诉模型每个词在序列中的位置

    

    """

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
    check_cuda_torch_info()
    # 先激活环境，再运行python

