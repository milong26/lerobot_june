#!/usr/bin/env python3
"""
自动生成评估脚本的流水线
功能：从 ~/.cache/huggingface/lerobot/ep10/ 下的所有文件夹中读取 tasks.parquet，
      提取 task 描述，并生成对应的 .sh 脚本到 eval_sh 目录
"""

import os
import pandas as pd
from pathlib import Path

# 配置路径
EP10_DIR = Path.home() / ".cache" / "huggingface" / "lerobot" / "ep10"
EVAL_SH_DIR = Path("/home/qwe/jun/lerobot/work/test_half_resolution/eval_sh")

# 脚本模板
SCRIPT_TEMPLATE = '''#!/bin/bash

# 运行方式：
# 1. 先赋予执行权限（仅首次需要）：
#    chmod +x {script_path}
#
# 2. 直接运行：
#    ./{script_name}
#    或者：
#    bash {script_path}
#
# 3. 如果需要修改参数，直接编辑本脚本中的对应行即可
# ============================================================

# 切换到脚本所在目录
cd /home/qwe/jun/lerobot/work/test_half_resolution

# 运行主控制程序
/home/qwe/anaconda3/envs/lb_local/bin/python main_controller.py \\
    --robot.type=so100_follower \\
    --robot.port=/dev/ttyACM1 \\
    --robot.id=start_new_heihei_2 \\
    --robot.cameras="{{wrist: {{type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: MJPG}},top: {{type: intelrealsense, serial_number_or_name: 806312060427, width: 640, height: 480, fps: 30, use_depth: False}}}}" \\
    --task="{task}" \\
    --server_url=ws://10.10.16.19:9000 \\
    --fps=30 \\
    --recording.enable_recording=True \\
    --recording.record_dir=./debug_recorded_data \\
    --recording.skip_frames=1 \\
    --recording.save_images=False \\
    --wowskin.enabled=True \\
    --wowskin.port=/dev/ttyACM0 \\
    --num_loop=5 \\
    --action_steps=20
'''


def extract_task_from_parquet(meta_dir: Path) -> str | None:
    """从 tasks.parquet 文件中提取 task 描述"""
    tasks_parquet = meta_dir / "tasks.parquet"
    
    if not tasks_parquet.exists():
        print(f"  ⚠ 未找到 tasks.parquet: {tasks_parquet}")
        return None
    
    try:
        df = pd.read_parquet(tasks_parquet)
        # 获取索引中的 task 描述
        if df.index.name == "task" and len(df) > 0:
            task = df.index[0]
            return task
        else:
            print(f"  ⚠ tasks.parquet 格式不符合预期: {tasks_parquet}")
            return None
    except Exception as e:
        print(f"  ✗ 读取 tasks.parquet 失败: {e}")
        return None


def generate_script(folder_name: str, task: str, output_dir: Path) -> Path:
    """生成 shell 脚本文件"""
    script_name = f"{folder_name}.sh"
    script_path = output_dir / script_name
    
    # 填充模板
    script_content = SCRIPT_TEMPLATE.format(
        folder_name=folder_name,
        script_name=script_name,
        script_path=script_path,
        task=task
    )
    
    # 写入文件
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    
    # 添加执行权限
    os.chmod(script_path, 0o755)
    
    return script_path


def main():
    """主函数：遍历 ep10 目录下的所有文件夹，生成对应的评估脚本"""
    print(f"扫描目录: {EP10_DIR}\n")
    
    if not EP10_DIR.exists():
        print(f"✗ 目录不存在: {EP10_DIR}")
        return
    
    # 确保输出目录存在
    EVAL_SH_DIR.mkdir(parents=True, exist_ok=True)
    
    generated_count = 0
    skipped_count = 0
    
    # 遍历 ep10 下的所有文件夹
    for folder in sorted(EP10_DIR.iterdir()):
        if not folder.is_dir():
            continue
        
        print(f"处理文件夹: {folder.name}")
        meta_dir = folder / "meta"
        
        # 提取 task 描述
        task = extract_task_from_parquet(meta_dir)
        if task is None:
            skipped_count += 1
            continue
        
        print(f"  ✓ 找到 task: {task}")
        
        # 生成脚本
        script_path = generate_script(folder.name, task, EVAL_SH_DIR)
        print(f"  ✓ 生成脚本: {script_path}\n")
        generated_count += 1
    
    # 输出统计信息
    print("=" * 60)
    print(f"完成！共生成 {generated_count} 个脚本，跳过 {skipped_count} 个文件夹")
    print(f"脚本保存位置: {EVAL_SH_DIR}")


if __name__ == "__main__":
    main()