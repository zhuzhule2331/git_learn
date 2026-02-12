import subprocess
import sys

def get_installed_packages():
    """获取已安装的Python库及其版本（兼容系统Python）"""
    # 优先使用pip3，其次用pip
    pip_commands = ["pip3", "pip"]
    pip_cmd = None
    
    # 检测系统可用的pip命令
    for cmd in pip_commands:
        try:
            subprocess.check_call(
                [cmd, "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            pip_cmd = cmd
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    if not pip_cmd:
        print("错误：系统中未找到pip/pip3命令")
        return {}
    
    try:
        # 调用系统pip list命令
        result = subprocess.check_output(
            [pip_cmd, "list"],
            universal_newlines=True,
            stderr=subprocess.STDOUT
        )
        
        # 解析输出
        lines = result.strip().split('\n')[2:]
        packages = {}
        for line in lines:
            if line.strip():
                parts = line.split()
                pkg_name = ' '.join(parts[:-1])
                pkg_version = parts[-1]
                packages[pkg_name] = pkg_version
        
        return packages
    except subprocess.CalledProcessError as e:
        print(f"执行pip命令出错：{e.output.strip()}")
        return {}
    except Exception as e:
        print(f"未知错误：{str(e)}")
        return {}

def get_message_from_ros():
    a  =  344
    c = 5
    return a+c

# 调用并打印结果
if __name__ == "__main__":
    installed_pkgs = get_installed_packages()
    if installed_pkgs:
        print("已安装的Python库及版本：")
        print("-" * 50)
        # 按包名排序，对齐输出
        for name, version in sorted(installed_pkgs.items()):
            print(f"{name:<25} {version}")
    else:
        print("未获取到已安装的库信息")