"""
测试 Evo-1 服务器连接
"""
import asyncio
import json
import websockets
import numpy as np

async def test_connection():
    server_url = "ws://10.10.16.19:8001"
    
    print(f"连接到 {server_url}...")
    
    async with websockets.connect(server_url, max_size=100_000_000) as ws:
        print("已连接！")
        
        # 创建测试 payload
        # 3 张 320x240x3 的随机图像
        test_images = []
        for i in range(3):
            img = np.random.randint(0, 255, (320, 240, 3), dtype=np.uint8).tolist()
            test_images.append(img)
        
        payload = {
            "image": test_images,
            "state": [0.0] * 24,
            "prompt": "Test prompt",
            "image_mask": [1, 1, 0],
            "action_mask": [1, 1, 1, 1, 1, 1] + [0] * 18
        }
        
        print(f"发送测试 payload...")
        print(f"  - 图像数量: {len(payload['image'])}")
        print(f"  - 图像尺寸: 240x320x3")
        print(f"  - state: 24 维")
        
        await ws.send(json.dumps(payload))
        print("已发送，等待响应...")
        
        try:
            response = await ws.recv()
            actions = json.loads(response)
            print(f"收到响应！动作数量: {len(actions)}")
            print(f"第一个动作: {actions[0][:6]}...")
        except Exception as e:
            print(f"错误: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())