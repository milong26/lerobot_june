# 要求
服务器推理（不是policy_server）本地执行代码运行
其中图片和state需要额外处理。

没懂隔4步是什么意思？

# 修改版本
## v1
65b725d5ccd772dd1594f888128a4fd4065e94bc

初始代码，什么都没修改

## v2
### 数据处理流程
work/test_half_resolution/mt50_evo1_client_prompt_ORG.py

参考代码里面是这样的：
1. 得到缓存数据（已经归一化了）
2. 处理state，image
3. 发送pyload
4. 得到action

那新的代码应该是
1. 得到state、force，拼接成state+force的形式
2. 处理image变成640*240
3. 发送pyload
4. 得到action


### 新增文件清单

#### 1. `data_collector.py` - 数据采集模块
**功能**：
- 连接机器人硬件（复用 LeRobot 的 `make_robot_from_config`）
- 实时读取摄像头图像（top + wrist）、关节状态（6维）、力传感器数据（15维）
- 维护历史数据缓冲区（deque，默认保存最近5帧）
- **异步数据录制**（后台线程，不阻塞主循环）：
  - 跳帧策略：每N帧保存1帧（默认3，降低CPU压力）
  - 图像降采样：640×480 → 160×120（1/16大小，节省空间）
  - 保存格式：npz（state/force/action）+ npy（images）
  - 队列机制：队列满时丢弃当前帧，不影响主循环
- 提供 `get_observation()` 接口获取当前帧数据
- 提供 `send_action()` 接口执行动作

#### 2. `data_processor.py` - 数据处理模块
**功能**：
- 加载 `norm_stats.json` 归一化统计信息
- 实现反归一化：将机器人的原始数据转换为模型需要的物理空间值
- 维度填充：state(6维)→24维，force(15维)→24维
- **图像处理**：
  - 分辨率减半：640×480 → 320×240
  - 水平拼接：top视角（左）+ wrist视角（右）= 640×240
  - 转换为 uint8 列表格式（用于 JSON 传输）
- **状态处理**：state(6维) + force(15维) = 21维，反归一化后填充到24维
- 构建完整的 payload 字典（包含 image、state、action、prompt、mask 等）

#### 3. `ws_client.py` - WebSocket 通信客户端
**功能**：
- 建立与推理服务器的 WebSocket 连接（`ws://10.10.16.19:9000`）
- 发送 JSON payload
- 接收服务器返回的推理结果
- 解析响应：提取 `act`（预测动作）和 `sta`（预测状态）
- 错误处理和自动重连机制

#### 4. `main_controller.py` - 主控制循环
**功能**：
- 整合以上3个模块
- 实现主循环：采集→处理→发送→接收→执行
- 控制执行频率（默认30Hz）
- 将服务器返回的动作发送给机器人执行
- 异常处理和优雅退出（Ctrl+C）

### 数据流向

```
机器人硬件
    ↓ (get_observation)
data_collector.py
    ↓ (image, state, force)
data_processor.py
    ↓ (JSON payload)
ws_client.py  ←→  WebSocket  ←→  推理服务器(10.10.16.19:9000)
    ↓ (actions, states)
main_controller.py
    ↓ (send_action)
机器人硬件 (执行动作)
```

### 关键处理说明

1. **图像处理**：
   - 原始分辨率：640×480
   - 处理后：320×240（减半）
   - 拼接方式：top视角（左）+ wrist视角（右）= 640×240

2. **状态处理**：
   - 原始：state(6维) + force(15维) = 21维
   - 服务器期望：24维（填充0）
   - 反归一化后发送给服务器

3. **动作执行**：
   - 服务器返回：actions(24维) + states(24维)
   - 本地解析：state取前6维，force取6-21维
   - 执行第一个动作，更新历史缓冲区

### 执行示例

**使用真机配置（与 `.vscode/launch.json` 中采集配置一致）**：

```bash
cd /home/qwe/jun/lerobot/work/test_half_resolution

python main_controller.py \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: MJPG},top: {type: intelrealsense, serial_number_or_name: 806312060427, width: 640, height: 480, fps: 30, use_depth: False}}" \
    --robot.id=start_new_heihei_2 \
    --task="Grab the cross-shape equipment." \
    --server_url=ws://10.10.16.19:9000 \
    --fps=30
```

**可选参数**：
- `--max_steps`: 最大执行步数（默认无限循环）
- `--fps`: 控制频率（默认30Hz）

**数据录制参数**：
- `--enable_recording`: 启用数据录制（默认关闭）
- `--record_dir`: 录制数据保存目录（默认 `./recorded_data`）
- `--skip_frames`: 跳帧数，每N帧保存1帧（默认3，降低CPU压力）
- `--save_images`: 是否保存图像（默认关闭，因为占用空间大）

### 数据录制说明

**设计理念**：使用后台异步线程保存数据，不阻塞主控制循环，对CPU压力最小化。

**跳帧策略**：
- 默认每3帧保存1帧（30Hz → 10Hz保存频率）
- 可调整 `--skip_frames` 参数（如设为2或5）

**图像降采样**：
- 原始：640×480
- 保存：160×120（每4像素取1个，面积减少到1/16）
- 大幅降低存储空间和写入时间

**存储格式**：
```
recorded_data/
└── 20260727_143022/          # 时间戳目录
    ├── frame_000000.npz      # state(6) + force(15) + timestamp
    ├── frame_000003.npz      # 跳帧：每3帧保存一次
    ├── frame_000006.npz
    ├── frame_000000_images.npy  # 图像（如果启用 --save_images）
    └── ...
```

**CPU优化**：
1. 后台线程异步保存，不阻塞主循环
2. 队列限制100帧，满了直接丢弃（不影响实时控制）
3. 图像降采样到1/16大小
4. 使用 `np.savez_compressed` 压缩保存
5. 跳帧策略减少保存频率

## 测试清单
1. 设备连接
2. observation获取
3. state和image拼接
4. 不send_action，检查action的值

### 设备连接
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
sudo chmod 666 /dev/ttyACM2

