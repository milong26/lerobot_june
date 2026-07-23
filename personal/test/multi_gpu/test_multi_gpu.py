#!/usr/bin/env python
"""
安全的多GPU测试脚本 - 用于验证SmolVLA在GPU 0,2,3上的训练环境

特点：
- 只运行5步，快速验证
- 小batch size，减少显存压力
- 超时保护机制，防止GPU卡住
- 模拟真实的SmolVLA+EP10配置
"""

import os
import subprocess
import sys
import signal
import shutil
from datetime import datetime

# 配置：使用GPU 0,2,3（避开1和4）
AVAILABLE_GPUS = "0,2,3"
NUM_PROCESSES = 3  # 3个GPU
TIMEOUT_SECONDS = 300  # 5分钟超时保护


def create_accelerate_config() -> str:
    """创建accelerate配置文件"""
    # 使用项目目录下的accelerate_tmp文件夹
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accelerate_tmp")
    os.makedirs(config_dir, exist_ok=True)
    
    config_path = os.path.join(config_dir, "accelerate_config.yaml")
    
    config_content = """compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: 'no'
num_processes: {num_processes}
use_cpu: false
gpu_ids: all
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
num_machines: 1
rdzv_backend: static
same_network: true
""".format(num_processes=NUM_PROCESSES)
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_path


def build_training_command(config_path: str, output_dir: str) -> list:
    """构建训练命令"""
    cmd = [
        "accelerate", "launch",
        "--config_file", config_path,
        "-m", "lerobot.scripts.lerobot_train",
        # 数据集配置
        "--dataset.repo_id=ep10/all",
        "--dataset.root=personal/work1/ep10_all",
        "--dataset.episodes=[0]",  # 只用第0个episode，快速测试
        # 策略配置
        "--policy.path=lerobot/smolvla_base",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        # 训练配置
        "--output_dir=" + output_dir,
        "--job_name=smolvla_multigpu_safety_test",
        "--steps=5",  # 只运行5步！
        "--batch_size=4",  # 小batch size
        "--eval_freq=-1",  # 禁用评估
        "--log_freq=1",  # 每步都日志
        "--save_freq=-1",  # 不保存checkpoint
        "--seed=42",
        "--num_workers=0",  # 减少数据加载复杂度
        # 重命名映射（与你的真实训练一致）
        '--rename_map={"observation.images.side":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}',
        # 禁用wandb（测试不需要）
        "--wandb.enable=false",
    ]
    return cmd


def run_with_timeout(cmd: list, env_vars: dict, timeout: int) -> tuple:
    """带超时保护的命令执行"""
    print(f"\n{'='*80}")
    print(f"开始多GPU安全测试")
    print(f"使用GPU: {AVAILABLE_GPUS}")
    print(f"进程数: {NUM_PROCESSES}")
    print(f"超时时间: {timeout}秒")
    print(f"{'='*80}\n")
    
    print("执行命令:")
    print(" ".join(cmd))
    print(f"\n环境变量 CUDA_VISIBLE_DEVICES={env_vars.get('CUDA_VISIBLE_DEVICES', 'N/A')}\n")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env_vars,
            preexec_fn=os.setsid  # 创建新的进程组，方便超时后清理
        )
        
        # 实时输出日志
        output_lines = []
        start_time = datetime.now()
        
        while True:
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                continue
            
            output_lines.append(line)
            # 实时打印到终端
            print(line, end='', flush=True)
            
            # 检查超时
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout:
                print(f"\n\n⚠️  超时！已运行 {elapsed:.0f} 秒，超过限制 {timeout} 秒")
                print("正在终止进程...")
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                return False, output_lines, "TIMEOUT"
        
        # 等待进程结束
        returncode = process.wait()
        
        if returncode == 0:
            print(f"\n{'='*80}")
            print("✅ 测试成功完成！")
            print(f"{'='*80}")
            return True, output_lines, "SUCCESS"
        else:
            print(f"\n{'='*80}")
            print(f"❌ 测试失败，返回码: {returncode}")
            print(f"{'='*80}")
            return False, output_lines, f"FAILED_WITH_CODE_{returncode}"
            
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")
        return False, [], f"EXCEPTION: {str(e)}"


def main():
    """主函数"""
    print("\n" + "="*80)
    print("SmolVLA 多GPU安全测试")
    print("="*80)
    
    # 检查CUDA可用性
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA不可用！")
            sys.exit(1)
        
        print(f"\n检测到 {torch.cuda.device_count()} 个GPU:")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("❌ PyTorch未安装！")
        sys.exit(1)
    
    # 创建输出目录路径（但不创建目录，让lerobot自己创建）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/test_smolvla_multigpu_safe_{timestamp}"
    # 注意：不要在这里创建目录，lerobot训练会自己创建
    # 如果目录已存在（比如上次测试失败留下的），删除它
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # 创建accelerate配置
    config_path = create_accelerate_config()
    print(f"\n✓ Accelerate配置已创建: {config_path}")
    
    # 构建命令
    cmd = build_training_command(config_path, output_dir)
    
    # 设置环境变量
    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = AVAILABLE_GPUS
    env_vars["HF_HUB_OFFLINE"] = "1"  # 离线模式
    
    # 关键修复：确保LD_LIBRARY_PATH正确传递给子进程
    # 这是torchcodec找到FFmpeg库的关键
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        conda_lib = os.path.join(conda_prefix, "lib")
        existing_ld = env_vars.get("LD_LIBRARY_PATH", "")
        if existing_ld:
            env_vars["LD_LIBRARY_PATH"] = f"{conda_lib}:{existing_ld}"
        else:
            env_vars["LD_LIBRARY_PATH"] = conda_lib
        print(f"\n✓ LD_LIBRARY_PATH 已设置为: {env_vars['LD_LIBRARY_PATH']}")
    
    # 执行测试
    success, output, status = run_with_timeout(cmd, env_vars, TIMEOUT_SECONDS)
    
    # 保存日志（此时训练应该已创建了output_dir）
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "test_log.txt")
    with open(log_file, 'w') as f:
        f.write(f"测试状态: {status}\n")
        f.write(f"使用GPU: {AVAILABLE_GPUS}\n")
        f.write(f"进程数: {NUM_PROCESSES}\n")
        f.write("="*80 + "\n")
        f.writelines(output)
    
    print(f"\n✓ 日志已保存到: {log_file}")
    
    # 返回状态
    if success:
        print("\n🎉 多GPU环境验证通过！可以安全进行正式训练。")
        print(f"\n正式训练命令示例:")
        print(f"""
nohup env CUDA_VISIBLE_DEVICES={AVAILABLE_GPUS} HF_HUB_OFFLINE=1 \\
accelerate launch \\
  --multi_gpu \\
  --num_processes={NUM_PROCESSES} \\
  $(which lerobot-train) \\
  --dataset.repo_id=ep10/all \\
  --dataset.root=personal/work1/ep10_all \\
  --policy.path=lerobot/smolvla_base \\
  --output_dir=outputs/test_smolvla_ep10 \\
  --job_name=smolvla_ep10 \\
  --policy.device=cuda \\
  --policy.push_to_hub=false \\
  --wandb.enable=true \\
  --steps=1000 \\
  --batch_size=32 \\
  --rename_map='{{"observation.images.side":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}}' \\
  > train_log/test_smolvla_ep10.log 2>&1 &
""")
        return 0
    else:
        print(f"\n⚠️  测试失败，状态: {status}")
        print("请检查日志文件以获取详细信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())