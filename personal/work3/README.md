# SIC Framework for MetaWorld

State Information Coverage (SIC) 框架在 MetaWorld 仿真环境中的实现。

## 项目目标

利用冻结的 VLM（SmolVLM2）嵌入空间，在采集任何训练数据之前，定量评估任意示范数据采集方案的状态覆盖质量，并用前向贪心算法在预算约束下输出最优采集清单。

## 快速开始

### 一键运行全流程

```bash
cd /data/zhonglinye/jun/lerobot
python personal/work3/run_full_pipeline.py --device cuda:0 --num-episodes 72 --budget_B 144
```

这会依次执行：
1. **数据生成**: 使用 Grid 策略生成 72 个 episode 的 B0 数据集（9 位置 × 8 姿态）
2. **嵌入提取**: 用冻结 SmolVLM2 提取全局视角和腕部视角的视觉嵌入（带缓存）
3. **锚点构建**: 拟合 PCA 并构建锚点参考系
4. **SIC 计算**: 计算 B0 的 SIC 分数并运行贪心规划
5. **图表生成**: 生成所有分析图片到 `personal/work3/figures/`
6. **报告生成**: 生成 `analysis_report.md` 解释所有图片

### 分步运行

```bash
# 1. 仅生成数据
python personal/work3/generate_dataset.py \
  --task pick-place-v3 \
  --strategy grid \
  --num-episodes 72 \
  --output-dir personal/work3/data/b0_dataset \
  --repo-id test/sic_b0 \
  --save-config-map personal/work3/data/config_map.json

# 2. 仅运行分析（需要数据集已存在）
python personal/work3/run_all.py \
  --dataset_root personal/work3/data/b0_dataset \
  --config_map personal/work3/data/config_map.json \
  --device cuda:0 \
  --budget_B 144

# 3. 仅重新生成图表（需要中间结果已存在）
python personal/work3/run_all.py \
  --mode figures_only
```

## 目录结构

```
personal/work3/
├── run_full_pipeline.py      # 一键运行全流程入口
├── run_all.py                # 分析管道入口
├── generate_dataset.py       # 数据生成脚本
├── configs.py                # 全局配置
├── sic/                      # SIC 核心模块
│   ├── embeddings.py         # 嵌入提取（带缓存）
│   ├── anchor.py             # 锚点参考系构建
│   ├── score.py              # SIC 分数计算
│   └── greedy.py             # 前向贪心规划算法
├── research/                 # 研究性分析
│   ├── embedding_analysis.py # 嵌入空间分析（t-SNE/PCA/距离曲线）
│   └── attention_analysis.py # 注意力图分析（需要训练好的模型）
├── visualize/
│   └── all_figures.py        # 所有图表生成函数
├── data/
│   ├── config_map.json       # episode → (position, rotation) 映射
│   ├── success_rates.json    # 各子集训练成功率（手动填入）
│   └── b0_dataset/           # B0 数据集（LeRobot 格式）
├── figures/                  # 输出的所有图片（PDF + PNG）
├── results/                  # 数值结果（JSON/NPY/PKL）
├── cache/                    # 嵌入缓存（避免重复计算）
├── logs/                     # 运行日志
└── analysis_report.md        # 自动生成的分析报告
```

## 生成的图表说明

| 图表 | 文件名 | 用途 |
|------|--------|------|
| Fig 1 | fig1_tsne_embeddings | t-SNE 可视化：验证嵌入空间聚类结构 |
| Fig 2 | fig2_cross_config_distance | 跨配置距离曲线：支撑锚点设计 |
| Fig 3 | fig3_pca_variance | PCA 方差曲线：证明 d_pca=32 合理 |
| Fig 4 | fig4_sic_correlation | SIC-成功率相关性（需要 success_rates.json） |
| Fig 5 | fig5_greedy_sic_curve | 贪心规划 SIC 增长曲线 |
| Fig 6 | fig6_collection_heatmap | 推荐采集方案热力图 |
| Fig 7 | fig7_baseline_comparison | 基线对比（需要 success_rates.json） |
| Fig 8 | fig8_efficiency_curve | 数据效率曲线 |
| Fig 9 | fig9_ablation_components | SIC 组件消融实验 |

## 硬件要求

- GPU: 至少 1 张 RTX 3090（24GB VRAM）
- 内存: 32GB+
- 磁盘: 50GB+（用于模型下载和数据集）

## 依赖

- Python 3.10+
- PyTorch
- transformers
- scikit-learn
- matplotlib
- seaborn
- LeRobot
- MetaWorld
- MuJoCo

## 配置说明

### config_map.json 格式

```json
{
  "0": [0, 0],
  "1": [0, 1],
  ...
  "71": [8, 7]
}
```

键是 episode_index（字符串），值是 `[position_id, rotation_id]`。

### success_rates.json 格式

```json
{
  "SIC-Guided-144": {"sic_score": 61.27, "success_rate": 93, "std": 1.0, "n_demos": 144},
  "Uniform-144": {"sic_score": 58.45, "success_rate": 84, "std": 0.8, "n_demos": 144}
}
```

## 注意事项

1. 首次运行会自动下载 SmolVLM2 模型（约 2GB）
2. 嵌入提取较慢，结果会自动缓存到 `cache/` 目录
3. 如果数据集已存在，使用 `--skip-generate` 跳过生成步骤
4. 成功率数据需要手动训练后填入 `success_rates.json`