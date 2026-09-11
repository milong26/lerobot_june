


Supported methods:




# 代码
之后
解析参数，然后创建目录，生成脚本，启动tmux并执行
执行过程：
选择episodes
训练过程：

以tinyvla_s为例

参数	值
Policy 类型	tinyvla_s
GPU	CUDA_VISIBLE_DEVICES=0
训练步数	50,000 steps
Batch Size	4
学习率	峰值 2e-4，warmup 250 steps，decay 50,000 steps 到 2.5e-6
保存频率	每 2000 steps 保存 checkpoint
环境评估频率	每 2000 steps
日志频率	每 200 steps
环境	MetaWorld disassemble-v3
相机	corner + gripperPOV
输出目录	.../checkpoints/
WandB	启用（用于实验追踪）


tinyvla微调方法说明（tinyvla-s举例）

加载lesjie/Llava-Pythia-400M 这个vlm，组成：Vision Tower+Multi-modal Projector (mm_projector)+Language Model (LLM)

TinyVLA 在 LLaVA-Pythia 的基础上添加了 Action Head
[图像输入] → Vision Tower → 视觉特征
                                    ↓
[语言指令] → Tokenizer → 文本嵌入 → Multi-modal Projector → LLM → embed_out → proj_to_action → [动作输出]
                                    ↑
[机器人状态] → 拼接进 LLM 输入
action head的类型是DROID Diffusion，训练时：给真实动作逐步加噪声，模型学习预测噪声，推理时：从纯噪声开始，逐步去噪生成动作序列。推理时使用 DDIM（Denoising Diffusion Implicit Models），默认 10 步去噪。

另外lora adapter 也参与微调。

它的动作分块一次预测16step

┌─────────────────────────────────────────────────────────────────┐
│                        DataLoader                               │
│  从 LeRobotDataset 中读取原始数据：                                │
│  - observation.images.corner: (B, C, H, W), dtype=uint8        │
│  - observation.images.gripperPOV: (B, C, H, W), dtype=uint8    │
│  - observation.state: (B, 7), 原始物理单位                        │
│  - action: (B, 16, 10), 原始物理单位                             │
│  - task: 语言指令字符串                                           │
│  - action_is_pad: 填充掩码                                       │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│              训练循环中的预处理 (lerobot_train.py)                │
│                                                                 │
│  1. 图像 uint8 → float32:                                       │
│     batch[cam_key] = batch[cam_key].to(float32) / 255.0         │
│     将像素值从 [0, 255] 归一化到 [0, 1]                          │
│                                                                 │
│  2. 移除不需要的特征:                                             │
│     batch.pop("observation.environment_state")                  │
│     (由 --remove_features 参数指定)                              │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ★ Preprocessor ★                              │
│         (make_tinyvla_pre_post_processors)                       │
│                                                                 │
│  输入处理管道 (input_steps):                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Step 1: RenameObservationsProcessorStep                   │  │
│  │   - 重命名观测字段（TinyVLA 使用空映射，不做重命名）          │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │ Step 2: AddBatchDimensionProcessorStep                    │  │
│  │   - 确保数据有 batch 维度                                  │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │ Step 3: DeviceProcessorStep                               │  │
│  │   - 将所有数据移动到指定设备 (cuda:0)                       │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │ Step 4: NormalizerProcessorStep                           │  │
│  │   - 对 STATE 和 ACTION 进行归一化 (MEAN_STD)               │  │
│  │   - 对 VISUAL 不做处理 (IDENTITY)                          │  │
│  │   - 公式: x_normalized = (x - mean) / std                 │  │
│  │   - mean/std 来自整个数据集的统计信息                        │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Policy.forward()                           │
│                                                                 │
│  输入 (已预处理):                                                │
│  - images: [0, 1] 范围的 float32                                │
│  - states: 归一化后的机器人状态 (mean≈0, std≈1)                  │
│  - actions: 归一化后的动作 (mean≈0, std≈1)                       │
│  - input_ids: tokenized 语言指令                                │
│                                                                 │
│  模型内部还会对图像做进一步处理 (_process_images):                 │
│  - Resize 到 vision tower 期望的尺寸                             │
│  - 用 CLIP/SigLIP 的 mean/std 做归一化                           │
│  - 转换为模型权重的 dtype (如 bfloat16)                          │
│                                                                 │
│  输出:                                                          │
│  - loss: 扩散模型的噪声预测损失                                    │
│  - info: 其他指标字典                                            │
└─────────────────────────────────────────────────────────────────┘

# train
运行
bash personal/work2/differentvlm/tinyvla/launch_tinyvla-s.sh --gpu-id 1 --num-episodes 200 --dataset-name disassemble-v3_corner --method random --seed 42
等价于
--method random

bash: /data/zhonglinye/application/miniconda3/envs/lb_server/lib/libtinfo.so.6: no version information available (required by bash)
tmux: /data/zhonglinye/application/miniconda3/envs/lb_server/lib/libtinfo.so.6: no version information available (required by tmux)
Launched experiment: tinyvla_s_random_ep200_seed42
tmux session: tinyvla_s_random_ep200_s42_disassemble-v3_corner
Output dir: /data/zhonglinye/jun/lerobot/personal/work2/differentvlm/tinyvla/tinyvla_s_random_ep200_seed42_disassemble-v3_corner
Dataset: disassemble-v3_corner
Dataset root: /data/zhonglinye/jun/lerobot/personal/work2/dataset_view/disassemble-v3_corner
GPU: 1
Episodes: 200
Method: random

Monitor with: tmux attach -t tinyvla_s_random_ep200_s42_disassemble-v3_corner
Check logs: tail -f /data/zhonglinye/jun/lerobot/personal/work2/differentvlm/tinyvla/tinyvla_s_random_ep200_seed42_disassemble-v3_corner/logs/tinyvla_s_random_ep200_seed42.log



# eval
看wandb效果好像有成功率非0的了，虽然我只测试了10次


## 24k：成功率3%


export MUJOCO_GL=egl 
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1

lerobot-eval \
    --policy.path=personal/work2/differentvlm/tinyvla/tinyvla_s_random_ep200_seed42_disassemble-v3_corner/checkpoints/024000/pretrained_model \
    --env.type=metaworld \
    --env.task=disassemble-v3 \
    --env.camera_name=corner,gripperPOV \
    --env.use_self_mw=true \
    --eval.batch_size=8 \
    --eval.n_episodes=200 \
    --policy.device=cuda \
    --policy.use_amp=false \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    2>&1 | tee personal/work2/eval_model/eval_mw_tvvla_s_disassemble-v3_corner_200ep_24kcp.log

# 20k 0.5%
export MUJOCO_GL=egl 
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1

lerobot-eval \
    --policy.path=personal/work2/differentvlm/tinyvla/tinyvla_s_random_ep200_seed42_disassemble-v3_corner/checkpoints/020000/pretrained_model \
    --env.type=metaworld \
    --env.task=disassemble-v3 \
    --env.camera_name=corner,gripperPOV \
    --env.use_self_mw=true \
    --eval.batch_size=8 \
    --eval.n_episodes=200 \
    --policy.device=cuda \
    --policy.use_amp=false \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    2>&1 | tee personal/work2/eval_model/eval_mw_tvvla_s_disassemble-v3_corner_200ep_20kcp.log

# 30k:1.0%
export MUJOCO_GL=egl 
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1

lerobot-eval \
    --policy.path=personal/work2/differentvlm/tinyvla/tinyvla_s_random_ep200_seed42_disassemble-v3_corner/checkpoints/030000/pretrained_model \
    --env.type=metaworld \
    --env.task=disassemble-v3 \
    --env.camera_name=corner,gripperPOV \
    --env.use_self_mw=true \
    --eval.batch_size=8 \
    --eval.n_episodes=200 \
    --policy.device=cuda \
    --policy.use_amp=false \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    2>&1 | tee personal/work2/eval_model/eval_mw_tvvla_s_disassemble-v3_corner_200ep_30kcp.log

# 50k: 3.0%
export MUJOCO_GL=egl 
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=2
export MUJOCO_EGL_DEVICE_ID=2

lerobot-eval \
    --policy.path=personal/work2/differentvlm/tinyvla/tinyvla_s_random_ep200_seed42_disassemble-v3_corner/checkpoints/050000/pretrained_model \
    --env.type=metaworld \
    --env.task=disassemble-v3 \
    --env.camera_name=corner,gripperPOV \
    --env.use_self_mw=true \
    --eval.batch_size=8 \
    --eval.n_episodes=200 \
    --policy.device=cuda \
    --policy.use_amp=false \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    2>&1 | tee personal/work2/eval_model/eval_mw_tvvla_s_disassemble-v3_corner_200ep_50kcp.log