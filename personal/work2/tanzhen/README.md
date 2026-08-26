# 探针实验 (Tanzhen Probe Experiment)

> 探针 × LeRobot × LIBERO-Plus 最基础验证

## 实验目标

**当前目标**：诊断单任务微调模型在真实部署会遇到的维度上是否有弱点

**原始目标**（已调整）：验证"表征/物理状态空间中的覆盖空隙是否与LIBERO-Plus式单轴受控扰动下的成功率跌幅正相关"

### 实验设计调整说明

根据实际部署场景（单任务、固定摄像头），对扰动轴进行了以下调整：

- **废弃camera轴**：对固定摄像头的单任务部署没有诊断价值，测"相机视角能不能变"是在测现实里几乎不会发生的场景
- **保留noise轴**：真实传感器噪声/运动模糊这类退化在固定摄像头下仍会真实发生
- **新增layout轴**：场景/物体摆放变化是真实部署里会自然出现的变量

**判断标准**：只保留"对应实际单任务、固定摄像头部署场景里会自然发生的变化"的轴——判断依据是"这个扰动在真实部署时有没有可能真的发生"，不是"LIBERO-Plus论文里有没有这个轴"。

## 目录结构

```
tanzhen/
├── README.md                          # 本文件
├── configs/
│   └── probe_config.yaml              # 实验配置中心
├── perturb/
│   ├── noise_perturb.py               # 传感器噪声扰动
│   ├── layout_perturb.py              # 布局扰动（goal位置偏移）
│   └── camera_perturb.py              # 相机视角扰动（已废弃，代码保留但不使用）
├── probe/
│   ├── run_probe_rollout.py           # 探针评测主流程
│   └── coverage_gap.py                # 覆盖空隙计算
├── analysis/
│   └── correlate_and_report.py        # 相关性分析 + 报告生成
└── results/
    ├── probe_raw/                     # 原始rollout记录
    ├── figures/                       # 可视化图表
    ├── weakness_scores.json           # 脆弱度权重（供v4使用）
    └── REPORT.md                      # 自动生成的实验报告
```

## 复现步骤

### 1. 环境准备

确保已安装以下依赖：
- numpy
- scipy
- matplotlib
- opencv-python
- pyyaml
- mujoco (官方Python绑定)
- metaworld

### 2. 冒烟测试（快速验证pipeline）

```bash
cd /data/zhonglinye/jun/lerobot

# 运行探针rollout（每格点1次rollout）
python personal/work2/tanzhen/probe/run_probe_rollout.py \
    --config personal/work2/tanzhen/configs/probe_config.yaml \
    --smoke-test

# 计算覆盖空隙
python personal/work2/tanzhen/probe/coverage_gap.py \
    --subset-file personal/work2/duibi/random_42/subsets/random_112_seed42.json \
    --dataset-metadata personal/work2/dataset_view/pickplacev3/episode_initial_states.json \
    --output personal/work2/tanzhen/results/probe_raw/random_42_coverage_gaps.json

python personal/work2/tanzhen/probe/coverage_gap.py \
    --subset-file personal/work2/duibi/uniform_42/subsets/uniform_112_seed42.json \
    --dataset-metadata personal/work2/dataset_view/pickplacev3/episode_initial_states.json \
    --output personal/work2/tanzhen/results/probe_raw/uniform_42_coverage_gaps.json

# 生成报告
python personal/work2/tanzhen/analysis/correlate_and_report.py \
    --config personal/work2/tanzhen/configs/probe_config.yaml
```

### 3. 正式测试

```bash
# 运行探针rollout（每格点3次rollout）
python personal/work2/tanzhen/probe/run_probe_rollout.py \
    --config personal/work2/tanzhen/configs/probe_config.yaml

# 重新生成报告
python personal/work2/tanzhen/analysis/correlate_and_report.py \
    --config personal/work2/tanzhen/configs/probe_config.yaml
```

## 待核实项确认结果

### 1. Checkpoint可用性

**确认结果**：
- **random_42**: `personal/work2/duibi/random_42/random_112_seed42/checkpoints/000200/pretrained_model/` - 已确认存在且可用
- **uniform_42**: `personal/work2/duibi/uniform_42/uniform_112_seed42/checkpoints/last/pretrained_model/` - 已确认存在且可用

这轮先用 **random_42** 跑通pipeline，确认代码无误后再补uniform_42做横向对比。

### 2. Layout扰动方案

**选定方案**：方案A - 扰动goal位置（保持物体格点位置不变）

**实现方式**：修改 `env.goal` 和 `env._get_state_rand_vec`
- **L1**: 目标位置偏移 ±0.02m
- **L2**: 目标位置偏移 ±0.04m

**核实依据**：MetaWorld的pick-place-v3任务中，goal/托盘位置和物体初始位置是两个独立可控参数。

### 3. 历史结果处理

之前 `results/probe_raw/` 里的camera轴相关结果是在占位符/随机动作状态下生成的，已标记无效：
- `random_42_camera-L1.json` → `random_42_camera-L1_INVALID_CAMERA_AXIS.json`
- `random_42_camera-L2.json` → `random_42_camera-L2_INVALID_CAMERA_AXIS.json`
- `uniform_42_camera-L1.json` → `uniform_42_camera-L1_INVALID_CAMERA_AXIS.json`
- `uniform_42_camera-L2.json` → `uniform_42_camera-L2_INVALID_CAMERA_AXIS.json`

这些文件不参与本次分析，仅保留作为历史记录。

### 4. MuJoCo绑定方式

**确认结果**：使用 **mujoco官方Python绑定**（非mujoco-py）

证据：`collect_metaworld_dataset.py` 中使用 `import mujoco`，且通过 `env.model.cam_pos[cam_id]` 访问相机参数。

### 5. 112格点来源

**确认结果**：从subset文件读取
- `random_42`: `personal/work2/duibi/random_42/subsets/random_112_seed42.json`
- `uniform_42`: `personal/work2/duibi/uniform_42/subsets/uniform_112_seed42.json`

## 实验配置

详见 `configs/probe_config.yaml`，主要参数：
- 扰动轴：noise (L0/L1/L2), layout (L0/L1/L2)
- Rollout次数：冒烟测试1次，正式测试3次
- β值：1.0（脆弱度权重缩放系数）
- GPU设备：默认使用cuda:0（可通过config中的gpu.device_id修改）

## 结论判读规则

### 主要产出（轴间对比 + 空间弱点分布）

- **轴间跌幅对比**：比较noise和layout两个轴各自的平均成功率跌幅，报告"哪个轴跌得更多"
  - > **免责声明**：两个轴的L2强度不是严格对等强度（噪声模糊 vs 目标位置偏移），排序只是方向性参考，不是精确定量结论
- **空间弱点分布图**：112格点按x-y坐标铺开的成功率跌幅热力图，标出每个轴下弱点集中在哪块区域
  - 如果两个轴的弱区重合 → 可能和该区域训练数据覆盖不足有关（见附录gap分析）
  - 如果两个轴的弱区不重合 → 说明是两种独立的、和覆盖度无关的脆弱模式

### 附录：覆盖空隙相关性分析

当前只有一个checkpoint的初步观察，尚不能下验证通过/不通过的结论。需第二个checkpoint补上后，按原计划要求两个checkpoint都显著才算稳健。

- **验证通过**：ρ > 0.3 且 p < 0.05 在两个checkpoint上都成立
- **结果不稳健**：只在一个checkpoint上成立
- **不支持假设**：两个都不显著

## 与v4的集成

生成的 `results/weakness_scores.json` 可直接被 `our/variants/v4_probe_weighted_greedy.py` 读取使用。

脆弱度计算公式：`w(c) = 1 + β · fragility_norm(c)`，其中fragility_norm是noise和layout两个轴fragility的均值归一化后的结果。