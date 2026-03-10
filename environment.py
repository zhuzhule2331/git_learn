import sys
import os
import numpy as np

# ===================== 第一步：检测基础系统环境（必选） =====================
def print_separator(title):
    """打印分隔线，美化输出"""
    print("\n" + "="*80)
    print(f"📌 {title}")
    print("="*80)

def check_system_env():
    """检测系统基础环境（无依赖，必执行）"""
    print_separator("【1/4】系统基础环境检测")
    
    # 1. Python版本
    print(f"🔹 Python 版本: {sys.version.split()[0]}")
    print(f"🔹 Python 架构: {sys.platform} ({'64位' if sys.maxsize > 2**32 else '32位'})")
    print(f"🔹 当前工作目录: {os.getcwd()}")
    
    # 2. OpenCV（可选，出错不影响）
    try:
        import cv2
        print(f"\n🔹 OpenCV 版本: {cv2.__version__}")
        cv2_gpu_count = cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, 'cuda') else 0
        print(f"🔹 OpenCV CUDA 支持: {'✅' if cv2_gpu_count > 0 else '❌'}")
        if cv2_gpu_count > 0:
            print(f"  - OpenCV可用GPU数: {cv2_gpu_count}")
    except ImportError:
        print("⚠️  OpenCV未安装，跳过OpenCV检测")
    except Exception as e:
        print(f"⚠️  OpenCV检测异常: {str(e)[:80]}")

# ===================== 第二步：检测PyTorch & CUDA（可选） =====================
def check_pytorch_cuda():
    """检测PyTorch和CUDA环境（独立检测，无PyTorch则跳过）"""
    print_separator("【2/4】PyTorch & CUDA 环境检测")
    
    try:
        import torch
    except ImportError:
        print("❌ 未安装PyTorch，跳过PyTorch/CUDA检测")
        return
    
    # 1. PyTorch基础信息
    try:
        print(f"🔹 PyTorch 版本: {torch.__version__}")
        print(f"🔹 PyTorch 安装路径: {torch.__file__}")
        
        # 2. CUDA基础检测
        print(f"\n🔹 系统CUDA版本（PyTorch检测）: {torch.version.cuda if hasattr(torch.version, 'cuda') else '未知'}")
        cuda_available = torch.cuda.is_available()
        print(f"🔹 CUDA 是否可用: {cuda_available}")
        
        if cuda_available:
            # 3. GPU设备详情
            gpu_count = torch.cuda.device_count()
            print(f"🔹 可用GPU数量: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_capability = torch.cuda.get_device_capability(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  - GPU {i}: {gpu_name} | 算力: {gpu_capability} | 显存: {gpu_memory:.2f}GB")
            
            # 4. 当前GPU & cuDNN
            current_device = torch.cuda.current_device()
            print(f"🔹 当前默认GPU: {current_device} ({torch.cuda.get_device_name(current_device)})")
            print(f"🔹 cuDNN 版本: {torch.backends.cudnn.version() if hasattr(torch.backends.cudnn, 'version') else '未知'}")
        else:
            print("⚠️  CUDA不可用，PyTorch将使用CPU运行")
            print(f"🔹 CPU核心数（PyTorch）: {torch.get_num_threads()}")
    except Exception as e:
        print(f"❌ PyTorch检测异常: {str(e)[:100]}")

# ===================== 第三步：检测ONNX Runtime（可选，适配有无ONNX） =====================
def check_onnxruntime():
    """检测ONNX Runtime环境（独立检测，无ONNX则跳过）"""
    print_separator("【3/4】ONNX Runtime 环境检测")
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("❌ 未安装ONNX Runtime，跳过ONNX检测")
        return
    
    # 1. ONNX Runtime基础信息
    try:
        print(f"🔹 ONNX Runtime 版本: {ort.__version__}")
        
        # 2. 可用执行提供者
        available_providers = ort.get_available_providers()
        print(f"🔹 可用执行提供者: {available_providers}")
        
        # 3. GPU支持检测
        has_cuda = "CUDAExecutionProvider" in available_providers
        has_tensorrt = "TensorrtExecutionProvider" in available_providers
        print(f"🔹 CUDAExecutionProvider 支持: {'✅' if has_cuda else '❌'}")
        print(f"🔹 TensorrtExecutionProvider 支持: {'✅' if has_tensorrt else '❌'}")
        
        # 4. 轻量GPU推理测试（无模型文件，避免依赖）
        if has_cuda:
            try:
                # 用极简张量测试GPU调用（无需保存模型文件）
                test_input = np.random.randn(1, 3, 288, 288).astype(np.float32)
                # 创建空会话（仅验证提供者加载）
                sess_options = ort.SessionOptions()
                sess_options.log_severity_level = 3  # 关闭日志
                # 模拟加载（无实际模型，仅验证CUDA提供者）
                dummy_sess = ort.InferenceSession(
                    b'',  # 空模型
                    sess_options=sess_options,
                    providers=["CUDAExecutionProvider"],
                    provider_options=[{"device_id": 0}]
                )
                print("🔹 ONNX Runtime GPU调用测试: ✅ 基础验证通过")
            except Exception as e:
                print(f"⚠️  ONNX Runtime GPU测试失败（不影响基础使用）: {str(e)[:80]}")
    except Exception as e:
        print(f"❌ ONNX Runtime检测异常: {str(e)[:100]}")

# ===================== 第四步：环境汇总（必选） =====================
def env_summary():
    """汇总检测结果，给出核心结论"""
    print_separator("【4/4】环境检测汇总")
    
    # 1. PyTorch/CUDA汇总
    try:
        import torch
        torch_ok = True
        cuda_ok = torch.cuda.is_available()
    except:
        torch_ok = False
        cuda_ok = False
    
    # 2. ONNX汇总
    try:
        import onnxruntime as ort
        onnx_ok = True
        onnx_gpu_ok = "CUDAExecutionProvider" in ort.get_available_providers()
    except:
        onnx_ok = False
        onnx_gpu_ok = False
    
    # 3. 输出汇总
    print("📝 核心结论:")
    print(f"  - PyTorch 安装: {'✅' if torch_ok else '❌'}")
    print(f"  - PyTorch CUDA: {'✅' if cuda_ok else '❌'}")
    print(f"  - ONNX Runtime 安装: {'✅' if onnx_ok else '❌'}")
    print(f"  - ONNX Runtime GPU: {'✅' if onnx_gpu_ok else '❌'}")
    
    # 4. 建议
    print("\n💡 环境建议:")
    if not torch_ok:
        print("  - 如需使用PyTorch: pip install torch torchvision torchaudio")
    if torch_ok and not cuda_ok:
        print("  - 如需PyTorch GPU: 安装对应CUDA版本的PyTorch（参考官网）")
    if not onnx_ok:
        print("  - 如需ONNX CPU: pip install onnxruntime")
        print("  - 如需ONNX GPU: pip install onnxruntime-gpu")
    if onnx_ok and not onnx_gpu_ok:
        print("  - ONNX GPU不可用: 卸载onnxruntime，安装onnxruntime-gpu")

# ===================== 主函数：分步执行，出错不终止 =====================
def main():
    print("🚀 开始环境检测（PyTorch/CUDA/ONNX Runtime）")
    print(f"检测时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 分步执行，每个模块独立运行
    check_system_env()       # 必执行
    check_pytorch_cuda()     # 可选，无PyTorch则跳过
    check_onnxruntime()      # 可选，无ONNX则跳过
    env_summary()            # 必执行，汇总结果
    
    print("\n🎉 环境检测完成！")

if __name__ == "__main__":
    main()