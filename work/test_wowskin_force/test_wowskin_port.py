#!/usr/bin/env python
"""
WowSkin 力传感器端口检测脚本
功能：检测 /dev/ttyACM0, /dev/ttyACM1, /dev/ttyACM2 中哪个是 WowSkin 力传感器
"""

import time
import sys
import os
import atexit
from anyskin import AnySkinProcess


def cleanup_sensor(sensor):
    """强制清理传感器进程"""
    if sensor is not None and sensor.is_alive():
        try:
            # 使用 terminate 强制终止，而不是 join 等待
            sensor.terminate()
            sensor.join(timeout=1.0)  # 短暂等待确认
        except Exception:
            pass  # 忽略清理错误


def test_port(port: str) -> bool:
    """
    测试指定端口是否为 WowSkin 力传感器
    
    Returns:
        True: 端口可用且是 WowSkin 传感器
        False: 端口不可用或不是 WowSkin 传感器
    """
    print(f"测试 {port}...", end=" ", flush=True)
    
    sensor = None
    
    try:
        # 创建传感器对象
        sensor = AnySkinProcess(num_mags=5, port=port)
        sensor.start()
        time.sleep(2.0)  # 等待传感器稳定
        
        # 尝试采集数据
        data = sensor.get_data(num_samples=5)
        
        # 检查数据格式
        import numpy as np
        data_array = np.array(data)
        
        if data_array.ndim != 2 or data_array.shape[0] == 0:
            print("❌ 失败 (数据维度异常)")
            return False
        
        # 显示第一条数据示例（简化）
        first_sample = data_array[0, 1:]  # 跳过时间戳
        print(f"✅ 是 WowSkin 力传感器 (数据: {first_sample[:3]}...)")
        return True
        
    except Exception as e:
        print(f"❌ 失败 ({type(e).__name__})")
        return False
    finally:
        # 强制清理传感器进程
        cleanup_sensor(sensor)


def main():
    """主函数：检测所有可能的端口"""
    ports = ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2"]
    
    print("="*50)
    print("WowSkin 力传感器端口检测")
    print("="*50)
    print(f"将检测以下端口: {', '.join(ports)}")
    print()
    
    found_ports = []
    
    for port in ports:
        if test_port(port):
            found_ports.append(port)
        
        # 在端口之间添加短暂延迟
        time.sleep(0.5)
    
    # 输出检测结果
    print(f"\n{'='*50}")
    print("检测结果:")
    print(f"{'='*50}")
    
    if found_ports:
        print(f"✅ WowSkin 力传感器端口: {', '.join(found_ports)}")
        print(f"\n设置权限命令:")
        for port in found_ports:
            print(f"  sudo chmod 666 {port}")
    else:
        print("❌ 未找到 WowSkin 力传感器")
        print("请检查连接和端口权限")
    
    print(f"{'='*50}")
    
    # 使用 os._exit 绕过 atexit 处理，避免 AnySkinProcess 的 atexit 注册导致卡住
    os._exit(0)


if __name__ == "__main__":
    main()