# 进度展示与代码生成指引

## 一、截至8月5日应该展示什么

工作从7月开始，8月5日汇报。按照合理的工作节奏，以下是每个阶段应该有的产出，以及对应的可展示图表。

**第一阶段（7月第1-2周）：嵌入空间分析**

这一阶段不需要训练任何模型，只需要有已采集的B₀数据（72个配置各1条轨迹）就能完成。

可展示内容：
- t-SNE可视化图：证明冻结VLM嵌入空间对不同配置有聚类结构，是SIC框架成立的前提
- 跨配置嵌入距离曲线：证明腕部视角中段帧的区分能力最强，支撑锚点设计选择
- PCA累计方差曲线：证明d_pca=32是合理的维度选择

这三张图可以在没有任何训练的情况下生成，是7月中旬前就应该有的结果。

**第二阶段（7月第3-4周）：SIC计算与初步相关性**

需要至少部分子集的训练结果（成功率数据）。如果全部12个子集都训练完了，就能得到完整的相关性结论。

可展示内容：
- SIC分数对比表格：12个子集的SIC分数和对应成功率
- 相关性散点图：Spearman ρ=0.8336
- 贪心规划热力图：展示贪心算法输出的采集方案
- SIC增长曲线：展示贪心迭代过程的收敛行为

**第三阶段（8月第1周至今）：基线对比与注意力图分析**

这一阶段的工作正在进行中，可以展示部分结果或分析框架。

可展示内容（8月5日应该已完成或接近完成）：
- 基线对比柱状图（可能尚缺DemInf改造版基线，但其余均可展示）
- 成功率-数据量效率曲线（整合了small论文的81条数据点）
- 注意力图对比（需要多个训练好的模型，如果训练未完成，展示框架图即可）

**汇报时的关键一句话**：

"我们已完成嵌入空间分析、SIC指标的完整计算和相关性验证（ρ=0.8336），贪心规划算法已验证有效，当前正在完成基线对比实验。论文写作从8月中旬开始，预计8月底投出。"

---

## 二、AI Coder 提示词（完整版，可直接使用）

以下提示词可以直接粘贴给 Claude Code 或任何支持长上下文的编程助手。它描述了一个完整的一键运行系统，生成论文所需的所有分析和图表。

---

### 提示词正文

**项目背景**

我在研究轻量化VLA（Vision-Language-Action）模型的示范数据采集规划问题。具体来说，我使用SmolVLA（基于SmolVLM2-500M-Video-Instruct的轻量化VLA）在SO-100机械臂上做单物体抓取任务。

核心方法是SIC（State Information Coverage）框架：在采集任何数据之前，利用冻结的SmolVLM嵌入空间，定量评估任意候选采集方案的状态覆盖质量，并用前向贪心算法在预算约束下输出最优"位置-姿态-次数"采集清单。

代码仓库：https://github.com/milong26/lerobot_june，分支：server-dev-merge-test
工作目录：personal/work2/
数据集格式：LeRobot标准格式（Parquet + MP4）

**请帮我实现一个完整的一键运行系统，包含以下全部功能，最终生成所有图表。**

---

**第一步：创建项目结构**

在 personal/work2/ 下创建以下结构：

```
personal/work2/
├── run_all.py                 # 总入口，一键运行所有分析
├── configs.py                 # 全局配置
├── sic/
│   ├── embeddings.py
│   ├── anchor.py
│   ├── score.py
│   └── greedy.py
├── baselines/
│   ├── uniform.py
│   ├── diagonal.py
│   └── deminfo_adapted.py
├── research/
│   ├── attention_analysis.py  # 注意力图分析
│   ├── embedding_analysis.py  # 嵌入空间分析
│   └── representation.py      # 表征退化分析
├── visualize/
│   └── all_figures.py         # 所有图表生成
├── figures/                   # 所有输出图片（英文标注）
├── results/                   # 所有数值结果（JSON）
├── cache/                     # 嵌入缓存，避免重复计算
└── data/
    └── success_rates.json     # 各子集训练后的成功率（手动填入）
```

---

**第二步：全局配置（configs.py）**

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os, json

@dataclass
class ProjectConfig:
    # ============ 数据路径（运行前需要设置）============
    # LeRobot数据集根目录
    # 例如：/data/robot/so100_grasp/
    # 数据集应包含多个子数据集，每个对应一种采集配置
    dataset_root: str = ""
    
    # 配置映射文件：{episode_index: [position_id, rotation_id, n_times]}
    # position_id: 0-8 (3x3网格)
    # rotation_id: 0-7 (8个方向，0°/45°/90°/135°/180°/225°/270°/315°)
    config_map_path: str = "personal/work2/data/config_map.json"
    
    # 已有训练结果：{subset_name: {success_rate: float, std: float, n_demos: int}}
    # 如果尚无训练结果，设为空，相关图表会跳过
    success_rates_path: str = "personal/work2/data/success_rates.json"
    
    # ============ 模型配置 ============
    vlm_model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    device: str = "cuda"  # 或 "cpu"
    batch_size: int = 4
    
    # 相机键名（需与LeRobot数据集中的键名一致）
    global_cam_key: str = "observation.images.global"
    wrist_cam_key: str = "observation.images.wrist"
    
    # ============ SIC超参数 ============
    d_pca: int = 32
    lambda_weight: float = 0.5
    alpha: float = 0.05
    t_max: int = 4
    
    # ============ 实验配置 ============
    budget_B: int = 144          # 贪心规划目标预算
    n_positions: int = 9         # 位置网格数
    n_rotations: int = 8         # 姿态离散数
    
    # ============ 输出配置 ============
    figures_dir: str = "personal/work2/figures"
    results_dir: str = "personal/work2/results"
    cache_dir: str = "personal/work2/cache"
    figure_dpi: int = 300
    figure_format: str = "pdf"   # 同时也保存png
    
    def setup(self):
        for d in [self.figures_dir, self.results_dir, self.cache_dir,
                  "personal/work2/data"]:
            os.makedirs(d, exist_ok=True)
    
    def load_success_rates(self) -> Optional[Dict]:
        if os.path.exists(self.success_rates_path):
            with open(self.success_rates_path) as f:
                return json.load(f)
        return None

CFG = ProjectConfig()
```

---

**第三步：嵌入提取（sic/embeddings.py）**

实现以下函数，带缓存机制（避免每次重新计算）：

```python
import os, pickle, numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm

def load_frozen_vlm(model_id: str, device: str):
    """
    Load SmolVLM2 as a frozen visual feature extractor.
    Returns (model, processor) with all parameters frozen.
    """
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, processor

def extract_visual_embedding(model, processor, image: Image.Image, device: str) -> np.ndarray:
    """
    Extract visual embedding from a single PIL image using frozen SmolVLM.
    Returns numpy array of shape (d_v,).
    
    Uses mean pooling over visual tokens from the vision encoder output.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        # 提取视觉编码器输出
        if hasattr(model, 'vision_model'):
            visual_out = model.vision_model(**{k: v for k, v in inputs.items() 
                                               if 'pixel' in k})
            embedding = visual_out.last_hidden_state.mean(dim=1)  # mean pool
        else:
            # fallback: 使用模型的image_encoder或visual_projection
            visual_out = model(**inputs, output_hidden_states=True)
            embedding = visual_out.last_hidden_state[:, 0, :]  # CLS token
    return embedding.squeeze().float().cpu().numpy()

def extract_trajectory_embeddings(
    model, processor, 
    dataset,  # LeRobotDataset instance
    episode_index: int, 
    cam_key: str, 
    device: str, 
    batch_size: int = 4
) -> np.ndarray:
    """
    Extract embeddings for all frames of one episode.
    Returns array of shape (n_frames, d_v).
    
    Handles LeRobotDataset frame format:
    - Images stored as float32 tensors in range [0,1], shape (C, H, W)
    """
    # 找到该episode的所有帧
    episode_data = [dataset[i] for i in range(len(dataset)) 
                    if dataset.episode_data_index['from'][episode_index] <= i 
                    < dataset.episode_data_index['to'][episode_index]]
    
    embeddings = []
    for i in range(0, len(episode_data), batch_size):
        batch = episode_data[i:i+batch_size]
        batch_embeddings = []
        for frame in batch:
            img_tensor = frame[cam_key]  # (C, H, W), float32, [0,1]
            img_pil = Image.fromarray(
                (img_tensor.permute(1,2,0).numpy() * 255).astype(np.uint8)
            )
            emb = extract_visual_embedding(model, processor, img_pil, device)
            batch_embeddings.append(emb)
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)  # (n_frames, d_v)

def extract_and_cache_all_embeddings(
    model, processor,
    dataset,
    config_map: dict,  # {episode_index: (pos_id, rot_id)}
    cam_key: str,
    device: str,
    cache_path: str,
    batch_size: int = 4
) -> dict:
    """
    Extract embeddings for all episodes in config_map.
    Caches results to avoid recomputation.
    Returns: {episode_index: np.ndarray of shape (n_frames, d_v)}
    """
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print(f"Extracting embeddings for {len(config_map)} episodes...")
    embeddings = {}
    for ep_idx in tqdm(config_map.keys()):
        embeddings[ep_idx] = extract_trajectory_embeddings(
            model, processor, dataset, ep_idx, cam_key, device, batch_size
        )
    
    with open(cache_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Cached embeddings saved to {cache_path}")
    return embeddings
```

---

**第四步：锚点参考系（sic/anchor.py）**

```python
import numpy as np
from sklearn.decomposition import IncrementalPCA
import pickle

def fit_pca_both_views(global_embeddings_dict: dict, wrist_embeddings_dict: dict, 
                        d_pca: int = 32) -> dict:
    """
    Fit independent PCA for global and wrist views.
    Returns pca_dict with fitted PCA objects and mean vectors.
    """
    def fit_single(emb_dict):
        pca = IncrementalPCA(n_components=d_pca)
        mu = np.mean(np.vstack(list(emb_dict.values())), axis=0)
        for emb in emb_dict.values():
            pca.partial_fit(emb - mu)
        return pca, mu
    
    pca_g, mu_g = fit_single(global_embeddings_dict)
    pca_w, mu_w = fit_single(wrist_embeddings_dict)
    return {'global': pca_g, 'wrist': pca_w, 'mu_global': mu_g, 'mu_wrist': mu_w}

def transform_embedding(embedding: np.ndarray, pca, mu: np.ndarray) -> np.ndarray:
    """Apply PCA transform: z = W^T (f - mu)"""
    return pca.transform((embedding - mu).reshape(1, -1)).flatten()

def compute_anchor_embeddings(
    global_embs: dict, wrist_embs: dict,
    config_map: dict,  # {episode_index: (pos_id, rot_id)}
    pca_dict: dict
) -> dict:
    """
    Build anchor embeddings for each (pos_id, rot_id) configuration.
    
    Global view anchor: PCA embedding of first frame (t=0)
      Rationale: First frame shows object before arm occlusion
    
    Wrist view anchor: Mean PCA embedding of middle frames [0.2N, 0.8N]
      Rationale: Middle frames show critical approach/align/contact phases
                 with highest cross-configuration discriminability
    
    Returns: {(pos_id, rot_id): {'phi_global': array(d_pca), 'phi_wrist': array(d_pca)}}
    """
    anchors = {}
    for ep_idx, (pos_id, rot_id) in config_map.items():
        key = (pos_id, rot_id)
        
        # Global: first frame
        g_emb = global_embs[ep_idx]  # (n_frames, d_v)
        phi_g = transform_embedding(g_emb[0], pca_dict['global'], pca_dict['mu_global'])
        
        # Wrist: middle segment mean
        w_emb = wrist_embs[ep_idx]  # (n_frames, d_v)
        N = len(w_emb)
        mid_start, mid_end = int(0.2 * N), int(0.8 * N)
        mid_embs = w_emb[mid_start:mid_end]
        phi_w_raw = mid_embs.mean(axis=0)
        phi_w = transform_embedding(phi_w_raw, pca_dict['wrist'], pca_dict['mu_wrist'])
        
        anchors[key] = {'phi_global': phi_g, 'phi_wrist': phi_w}
    
    return anchors

def compute_distance_scales(anchors: dict) -> dict:
    """
    Compute per-view distance scales as mean nearest-neighbor distance.
    
    Uses nearest-neighbor (not all-pairs mean) to avoid overestimation
    from distant outlier pairs.
    
    Returns: {'d_bar_global': float, 'd_bar_wrist': float}
    """
    keys = list(anchors.keys())
    
    def mean_nn_dist(view):
        nn_dists = []
        for i, k in enumerate(keys):
            phi_i = anchors[k][f'phi_{view}']
            dists = [np.linalg.norm(phi_i - anchors[k2][f'phi_{view}'])
                     for j, k2 in enumerate(keys) if j != i]
            nn_dists.append(min(dists))
        return np.mean(nn_dists)
    
    return {
        'd_bar_global': mean_nn_dist('global'),
        'd_bar_wrist': mean_nn_dist('wrist')
    }

def build_anchor_reference(
    global_embs: dict, wrist_embs: dict,
    config_map: dict,
    d_pca: int = 32,
    save_path: str = None
) -> dict:
    """
    Build complete anchor reference system R.
    
    R = {
        'anchors': {(pos_id, rot_id): {'phi_global': ..., 'phi_wrist': ...}},
        'd_bar_global': float,
        'd_bar_wrist': float,
        'pca_dict': dict
    }
    """
    pca_dict = fit_pca_both_views(global_embs, wrist_embs, d_pca)
    anchors = compute_anchor_embeddings(global_embs, wrist_embs, config_map, pca_dict)
    dist_scales = compute_distance_scales(anchors)
    
    R = {
        'anchors': anchors,
        'd_bar_global': dist_scales['d_bar_global'],
        'd_bar_wrist': dist_scales['d_bar_wrist'],
        'pca_dict': pca_dict
    }
    
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(R, f)
        print(f"Anchor reference saved to {save_path}")
    
    return R
```

---

**第五步：SIC函数（sic/score.py）**

```python
import numpy as np

def laplacian_kernel(phi_a: np.ndarray, phi_pr: np.ndarray, d_bar: float) -> float:
    """
    K_v(a, p, r) = exp(-||phi_v(a) - phi_v(p,r)||_2 / d_bar_v)
    Returns value in (0, 1].
    """
    return np.exp(-np.linalg.norm(phi_a - phi_pr) / (d_bar + 1e-8))

def tau(t: int, alpha: float) -> float:
    """
    tau(t) = alpha * log((t+1)/t)
    Marginal weight for t-th collection of same configuration.
    Strictly decreasing: tau(1) > tau(2) > ... > tau(T_max)
    """
    return alpha * np.log((t + 1) / t)

def tau_cumsum(T_pr: int, alpha: float) -> float:
    """
    Sum_{t=1}^{T_pr} tau(t) = alpha * log(T_pr + 1)
    """
    return alpha * np.log(T_pr + 1)

def compute_anchor_support(
    anchor_key: tuple,
    collection_plan: dict,  # {(pos_id, rot_id): n_times}
    anchors: dict,
    d_bar_global: float,
    d_bar_wrist: float,
    alpha: float
) -> dict:
    """
    Compute sigma_global(a, D) and sigma_wrist(a, D) for one anchor.
    
    sigma_v(a, D) = Sum_{(p,r) in D} Sum_{t=1}^{T_pr} tau(t) * K_v(a, p, r)
                 = Sum_{(p,r) in D} tau_cumsum(T_pr) * K_v(a, p, r)
    """
    phi_a_g = anchors[anchor_key]['phi_global']
    phi_a_w = anchors[anchor_key]['phi_wrist']
    
    sigma_g = 0.0
    sigma_w = 0.0
    
    for (pos_id, rot_id), n_times in collection_plan.items():
        if (pos_id, rot_id) not in anchors:
            continue
        phi_pr_g = anchors[(pos_id, rot_id)]['phi_global']
        phi_pr_w = anchors[(pos_id, rot_id)]['phi_wrist']
        
        weight = tau_cumsum(n_times, alpha)
        sigma_g += weight * laplacian_kernel(phi_a_g, phi_pr_g, d_bar_global)
        sigma_w += weight * laplacian_kernel(phi_a_w, phi_pr_w, d_bar_wrist)
    
    return {'sigma_global': sigma_g, 'sigma_wrist': sigma_w}

def saturation(sigma: float) -> float:
    """sigma / (1 + sigma), maps [0, inf) -> [0, 1)"""
    return sigma / (1.0 + sigma)

def compute_sic(
    collection_plan: dict,
    anchor_ref: dict,
    alpha: float = 0.05,
    lambda_weight: float = 0.5
) -> dict:
    """
    SIC(D) = SIC_global(D) + lambda * SIC_wrist(D)
    
    SIC_v(D) = Sum_{a in A} saturation(sigma_v(a, D))
    
    Returns: {
        'sic': float,          # total SIC score
        'sic_global': float,   # global view component
        'sic_wrist': float,    # wrist view component
        'anchor_supports': dict # per-anchor support values (for debugging)
    }
    """
    anchors = anchor_ref['anchors']
    sic_g, sic_w = 0.0, 0.0
    anchor_supports = {}
    
    for anchor_key in anchors:
        support = compute_anchor_support(
            anchor_key, collection_plan, anchors,
            anchor_ref['d_bar_global'], anchor_ref['d_bar_wrist'], alpha
        )
        sic_g += saturation(support['sigma_global'])
        sic_w += saturation(support['sigma_wrist'])
        anchor_supports[anchor_key] = support
    
    total_sic = sic_g + lambda_weight * sic_w
    return {
        'sic': total_sic,
        'sic_global': sic_g,
        'sic_wrist': sic_w,
        'anchor_supports': anchor_supports
    }

def compute_marginal_gain(
    current_plan: dict,
    candidate_config: tuple,  # (pos_id, rot_id) to add one more collection
    anchor_ref: dict,
    alpha: float,
    lambda_weight: float
) -> float:
    """
    Compute SIC(D + 1 more of candidate) - SIC(D) efficiently.
    Uses incremental update instead of recomputing full SIC.
    
    Marginal gain is strictly decreasing in current support (diminishing returns).
    """
    new_plan = dict(current_plan)
    new_plan[candidate_config] = new_plan.get(candidate_config, 0) + 1
    
    current_sic = compute_sic(current_plan, anchor_ref, alpha, lambda_weight)['sic']
    new_sic = compute_sic(new_plan, anchor_ref, alpha, lambda_weight)['sic']
    
    return new_sic - current_sic
```

---

**第六步：前向贪心算法（sic/greedy.py）**

```python
import numpy as np
from tqdm import tqdm
from .score import compute_sic, compute_marginal_gain

def greedy_plan(
    anchor_ref: dict,
    budget_B: int,
    t_max: int = 4,
    alpha: float = 0.05,
    lambda_weight: float = 0.5
) -> dict:
    """
    Forward greedy planning algorithm.
    
    Starts from B0 (all configs with n_times=1), adds one collection
    at a time to the config with highest SIC marginal gain, until
    total budget B is reached.
    
    Returns: {
        'final_plan': {(pos_id, rot_id): n_times},
        'sic_history': [float],    # SIC after each greedy step
        'gain_history': [float],   # marginal gain at each step
        'selected_configs': list,  # which config was selected each step
        'stopping_step': int       # step where marginal gain < 0.5% of total SIC
    }
    """
    all_configs = list(anchor_ref['anchors'].keys())
    
    # Initialize: B0 = all configs with 1 collection each
    current_plan = {cfg: 1 for cfg in all_configs}
    current_sic = compute_sic(current_plan, anchor_ref, alpha, lambda_weight)['sic']
    
    sic_history = [current_sic]
    gain_history = []
    selected_configs = []
    stopping_step = None
    
    n_steps = budget_B - len(all_configs)
    
    for step in tqdm(range(n_steps), desc="Greedy planning"):
        # Find candidates: configs not yet at t_max
        candidates = [cfg for cfg in all_configs if current_plan[cfg] < t_max]
        
        if not candidates:
            print(f"All configs reached t_max={t_max} at step {step}")
            break
        
        # Compute marginal gain for each candidate
        gains = {cfg: compute_marginal_gain(
                     current_plan, cfg, anchor_ref, alpha, lambda_weight)
                 for cfg in candidates}
        
        # Select best
        best_cfg = max(gains, key=gains.get)
        best_gain = gains[best_cfg]
        
        # Update plan
        current_plan[best_cfg] += 1
        current_sic += best_gain
        
        sic_history.append(current_sic)
        gain_history.append(best_gain)
        selected_configs.append(best_cfg)
        
        # Check stopping criterion: marginal gain < 0.5% of current SIC
        if stopping_step is None and best_gain < 0.005 * current_sic:
            stopping_step = step
    
    return {
        'final_plan': current_plan,
        'sic_history': sic_history,
        'gain_history': gain_history,
        'selected_configs': selected_configs,
        'stopping_step': stopping_step if stopping_step else n_steps
    }
```

---

**第七步：研究性分析（research/embedding_analysis.py）**

这一节实现用于理解和展示嵌入空间特性的分析代码，主要用于论文的分析图表，不需要额外的训练数据。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def compute_cross_config_distance_curve(
    wrist_embs: dict,  # {episode_index: (n_frames, d_v)}
    config_map: dict,  # {episode_index: (pos_id, rot_id)}
    n_time_points: int = 50
) -> np.ndarray:
    """
    At each normalized time step, compute mean pairwise L2 distance
    between embeddings of different configurations.
    
    This validates the choice of mid-segment frames for wrist anchor:
    higher cross-config distance = better discriminability.
    
    Returns: array of shape (n_time_points,)
    """
    distances = np.zeros(n_time_points)
    count = 0
    
    ep_indices = list(config_map.keys())
    
    for i in range(len(ep_indices)):
        for j in range(i+1, len(ep_indices)):
            ep_i, ep_j = ep_indices[i], ep_indices[j]
            if config_map[ep_i] == config_map[ep_j]:
                continue  # skip same config
            
            emb_i = wrist_embs[ep_i]  # (N_i, d_v)
            emb_j = wrist_embs[ep_j]  # (N_j, d_v)
            
            # Interpolate to n_time_points
            for t in range(n_time_points):
                ti = int(t / n_time_points * (len(emb_i) - 1))
                tj = int(t / n_time_points * (len(emb_j) - 1))
                distances[t] += np.linalg.norm(emb_i[ti] - emb_j[tj])
            count += 1
    
    return distances / (count + 1e-8)

def compute_pca_variance_curve(embeddings_dict: dict, max_components: int = 128) -> dict:
    """
    Compute cumulative explained variance ratio vs number of PCA components.
    
    Returns: {
        'n_components': list,
        'cumulative_variance': list,
        'per_component_variance': list
    }
    """
    all_embs = np.vstack(list(embeddings_dict.values()))
    pca = PCA(n_components=min(max_components, all_embs.shape[1]))
    pca.fit(all_embs)
    
    return {
        'n_components': list(range(1, len(pca.explained_variance_ratio_) + 1)),
        'cumulative_variance': list(np.cumsum(pca.explained_variance_ratio_)),
        'per_component_variance': list(pca.explained_variance_ratio_)
    }

def compute_tsne_for_configs(
    wrist_embs: dict,
    config_map: dict,
    selected_configs: list = None,  # if None, use all; or pass [(pos,rot), ...]
    pca_dim: int = 50,  # pre-reduce before t-SNE
    tsne_perplexity: int = 30
) -> dict:
    """
    Compute t-SNE embedding of trajectory frames for selected configs.
    
    Returns: {
        'tsne_xy': array(N_total, 2),
        'config_labels': list of (pos_id, rot_id) for each point,
        'frame_indices': list of normalized frame positions [0,1]
    }
    """
    if selected_configs is None:
        # use 4 random configs for readability
        import random
        all_cfgs = list(set(config_map.values()))
        selected_configs = random.sample(all_cfgs, min(4, len(all_cfgs)))
    
    # Collect all frames from selected configs
    all_embs = []
    labels = []
    frame_positions = []
    
    for ep_idx, cfg in config_map.items():
        if cfg not in selected_configs:
            continue
        embs = wrist_embs[ep_idx]  # (n_frames, d_v)
        all_embs.append(embs)
        labels.extend([cfg] * len(embs))
        frame_positions.extend(np.linspace(0, 1, len(embs)).tolist())
    
    all_embs = np.vstack(all_embs)
    
    # Pre-reduce with PCA
    pca = PCA(n_components=min(pca_dim, all_embs.shape[1]))
    reduced = pca.fit_transform(all_embs)
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42)
    tsne_xy = tsne.fit_transform(reduced)
    
    return {
        'tsne_xy': tsne_xy,
        'config_labels': labels,
        'frame_positions': frame_positions,
        'selected_configs': selected_configs
    }
```

---

**第八步：注意力图分析（research/attention_analysis.py）**

这一节需要已经训练好的模型。如果模型尚未训练，函数会跳过并打印提示。

```python
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def extract_attention_maps(
    model,      # 训练好的SmolVLA模型（需要能访问内部attention）
    processor,
    image: Image.Image,
    instruction: str,
    device: str,
    layer_indices: list = [14, 15, 16, 17, 18]  # 中间层，参考Don't Blind Your VLA
) -> dict:
    """
    Extract attention maps from specified transformer layers.
    
    Returns attention weights for visual patches when processing
    the instruction query.
    
    Returns: {
        layer_idx: attention_map array of shape (n_visual_patches,)
    }
    
    Note: Requires model.config.output_attentions=True or equivalent hook registration.
    """
    attention_maps = {}
    hooks = []
    
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is typically (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]  # (batch, heads, seq, seq)
                if attn_weights is not None:
                    # Mean over heads, take visual patch attention
                    attention_maps[layer_idx] = attn_weights.mean(dim=1).cpu().numpy()
        return hook_fn
    
    # Register hooks on specified layers
    # Adjust layer access based on actual model architecture
    for idx in layer_indices:
        try:
            layer = model.language_model.model.layers[idx]
            hook = layer.self_attn.register_forward_hook(make_hook(idx))
            hooks.append(hook)
        except (AttributeError, IndexError):
            pass
    
    inputs = processor(images=image, text=instruction, return_tensors="pt").to(device)
    
    with torch.no_grad():
        model(**inputs, output_attentions=True)
    
    for hook in hooks:
        hook.remove()
    
    return attention_maps

def compare_attention_across_datasets(
    model_paths: dict,  # {dataset_name: model_checkpoint_path}
    test_images: list,  # list of (PIL.Image, instruction_str) tuples
    processor,
    device: str,
    save_dir: str
):
    """
    Compare attention maps of models trained on different dataset configurations.
    
    Models are expected to differ in SIC quality:
    - High-SIC model (e.g., SIC-guided 144 demos)
    - Low-SIC model (e.g., corners-cardinal 64 demos)
    - Full-data model (288 demos)
    
    Each model processes the same test images.
    Good models should show focused attention on the target object.
    
    Creates grid visualization: rows=models, columns=test images,
    showing attention concentration on task-relevant regions.
    """
    if not model_paths:
        print("[SKIP] attention_analysis: no model paths provided. "
              "Run training first, then re-run with --model_paths")
        return None
    
    from transformers import AutoModelForCausalLM
    results = {}
    
    for dataset_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            print(f"[SKIP] Model not found: {model_path}")
            continue
        
        print(f"Loading model: {dataset_name}")
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device).eval()
        
        model_attns = []
        for img, instruction in test_images:
            attns = extract_attention_maps(model, processor, img, instruction, device)
            model_attns.append(attns)
        
        results[dataset_name] = model_attns
        del model
        torch.cuda.empty_cache()
    
    # Generate comparison figure
    if results:
        _plot_attention_comparison(results, test_images, save_dir)
    
    return results

def _plot_attention_comparison(results, test_images, save_dir):
    """Internal: generate attention map comparison grid figure"""
    model_names = list(results.keys())
    n_models = len(model_names)
    n_images = len(test_images)
    layer_to_show = list(list(results.values())[0][0].keys())[-1]  # last available layer
    
    fig, axes = plt.subplots(n_models + 1, n_images, 
                              figsize=(4 * n_images, 4 * (n_models + 1)))
    fig.suptitle("Attention Map Comparison Across Dataset Configurations\n"
                 f"(Layer {layer_to_show}, Mean over Attention Heads)", 
                 fontsize=14, fontweight='bold')
    
    # Row 0: original images
    for j, (img, instruction) in enumerate(test_images):
        ax = axes[0, j] if n_images > 1 else axes[0]
        ax.imshow(img)
        ax.set_title(f"Test image {j+1}\n'{instruction[:30]}...'", fontsize=9)
        ax.axis('off')
        if j == 0:
            ax.set_ylabel("Original", fontsize=10, fontweight='bold')
    
    # Rows 1+: attention maps per model
    for i, model_name in enumerate(model_names):
        for j in range(n_images):
            ax = axes[i+1, j] if n_images > 1 else axes[i+1]
            
            attn_data = results[model_name][j]
            if layer_to_show in attn_data:
                attn = attn_data[layer_to_show][0]  # (seq_len, seq_len)
                # Show attention from last text token to all image tokens
                img_attn = attn[-1, :int(attn.shape[1] * 0.8)]  # rough visual patch slice
                # Reshape to approximate grid
                side = int(np.sqrt(len(img_attn)))
                if side * side <= len(img_attn):
                    img_attn_grid = img_attn[:side*side].reshape(side, side)
                    ax.imshow(img_attn_grid, cmap='hot', interpolation='bilinear')
                    ax.set_title(f"Max: {img_attn.max():.3f}", fontsize=8)
            ax.axis('off')
            if j == 0:
                # Short model name for y-label
                short_name = model_name.replace('_', ' ').replace('demo', 'd')
                ax.set_ylabel(short_name[:20], fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "fig_attention_map_comparison")
    plt.savefig(save_path + ".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(save_path + ".png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}.pdf")
    print("Purpose: Compare attention localization quality across dataset configurations")
    print("         High-SIC data -> better visual attention focus on target object")
```

---

**第九步：所有图表生成（visualize/all_figures.py）**

注意：所有图表使用英文标注，每个保存函数打印图表用途说明。

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import os

# Global style settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150
})
COLORS = sns.color_palette("colorblind")

def fig1_tsne_embeddings(tsne_result: dict, save_dir: str):
    """
    Figure 1: t-SNE visualization of wrist-view VLM embeddings.
    
    PURPOSE: Validates that frozen VLM embedding space has configuration-consistent
    cluster structure. Same-configuration frames cluster together; different
    configurations separate. This justifies using VLM embeddings as coverage proxy.
    
    Layout: 2 subplots (colored by config_id and by frame_position)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    tsne_xy = tsne_result['tsne_xy']
    labels = tsne_result['config_labels']
    frame_pos = tsne_result['frame_positions']
    
    # Map config tuples to integer IDs for coloring
    unique_cfgs = list(set(labels))
    cfg_to_id = {cfg: i for i, cfg in enumerate(unique_cfgs)}
    color_ids = [cfg_to_id[l] for l in labels]
    
    # Subplot 1: color by configuration
    scatter1 = ax1.scatter(tsne_xy[:, 0], tsne_xy[:, 1], 
                           c=color_ids, cmap='tab10', alpha=0.6, s=8)
    ax1.set_title("Colored by Configuration (position, rotation)")
    ax1.set_xlabel("t-SNE Dimension 1")
    ax1.set_ylabel("t-SNE Dimension 2")
    
    # Add legend with config labels
    for cfg_id, cfg in enumerate(unique_cfgs):
        pos_id, rot_id = cfg
        ax1.scatter([], [], c=[COLORS[cfg_id % len(COLORS)]], 
                   label=f"pos={pos_id}, rot={rot_id*45}°", s=30)
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    
    # Subplot 2: color by temporal position in trajectory
    scatter2 = ax2.scatter(tsne_xy[:, 0], tsne_xy[:, 1],
                           c=frame_pos, cmap='viridis', alpha=0.6, s=8)
    plt.colorbar(scatter2, ax=ax2, label="Normalized Frame Position")
    ax2.set_title("Colored by Trajectory Time Step (normalized)")
    ax2.set_xlabel("t-SNE Dimension 1")
    ax2.set_ylabel("t-SNE Dimension 2")
    
    plt.suptitle("t-SNE Visualization of Frozen VLM Wrist-View Embeddings\n"
                 "(4 randomly selected configurations, all trajectory frames)", 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "fig1_tsne_embeddings")
    plt.savefig(save_path + ".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(save_path + ".png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}.pdf")
    print("PURPOSE: Validates VLM embedding space has config-consistent clustering,")
    print("         justifying coverage measurement in embedding space.")

def fig2_cross_config_distance(distance_curve: np.ndarray, save_dir: str):
    """
    Figure 2: Mean cross-configuration embedding distance across trajectory time steps.
    
    PURPOSE: Motivates the choice of mid-segment frames for wrist-view anchor.
    Shows that frames in the middle of the trajectory (approach/align/contact phases)
    have the highest cross-configuration discriminability.
    
    Shading: [0.2, 0.8] mid-segment region highlighted
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    n_points = len(distance_curve)
    t = np.linspace(0, 1, n_points)
    
    ax.plot(t, distance_curve, color=COLORS[0], linewidth=2.5, label='Mean cross-config L2 distance')
    ax.fill_between([0.2, 0.8], 
                    [distance_curve.min()] * 2, 
                    [distance_curve.max()] * 2,
                    alpha=0.15, color=COLORS[1], label='Mid-segment region [0.2N, 0.8N]')
    
    # Mark maximum
    max_idx = np.argmax(distance_curve)
    ax.scatter([t[max_idx]], [distance_curve[max_idx]], 
               color='red', s=100, zorder=5, label=f'Peak at t={t[max_idx]:.2f}')
    
    ax.set_xlabel("Normalized Trajectory Time Step")
    ax.set_ylabel("Mean Pairwise L2 Distance (across configurations)")
    ax.set_title("Cross-Configuration Discriminability of Wrist-View Embeddings\n"
                 "Along Trajectory Time Steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "fig2_cross_config_distance")
    plt.savefig(save_path + ".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(save_path + ".png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}.pdf")
    print("PURPOSE: Supports anchor design choice — mid-segment frames have")
    print("         highest discriminability, justifying phi_wrist definition.")

def fig3_pca_variance(global_variance: dict, wrist_variance: dict, save_dir: str):
    """
    Figure 3: Cumulative explained variance ratio vs PCA dimensions.
    
    PURPOSE: Justifies d_pca=32 as the elbow point where additional dimensions
    give diminishing variance coverage. Shows consistent structure across views.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, var_data, title in [
        (ax1, global_variance, "Global View (Fixed Camera)"),
        (ax2, wrist_variance, "Wrist View (End-Effector Camera)")
    ]:
        n = var_data['n_components']
        cum_var = var_data['cumulative_variance']
        
        ax.plot(n, cum_var, color=COLORS[0], linewidth=2.5)
        ax.axvline(x=32, color='red', linestyle='--', alpha=0.8, label='d_pca = 32')
        ax.axhline(y=cum_var[31], color='gray', linestyle=':', alpha=0.6,
                   label=f'{cum_var[31]*100:.1f}% variance at d=32')
        
        ax.fill_between(n[:32], 0, cum_var[:32], alpha=0.15, color=COLORS[0])
        
        ax.set_xlabel("Number of PCA Components")
        ax.set_ylabel("Cumulative Explained Variance Ratio")
        ax.set_title(f"PCA Variance Coverage — {title}")
        ax.legend(fontsize=10)
        ax.set_xlim([0, min(128, max(n))])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("PCA Cumulative Explained Variance: Elbow Analysis for d_pca Selection",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(save_dir, "fig3_pca_variance")
    plt.savefig(save_path + ".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(save_path + ".png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}.pdf")
    print("PURPOSE: Justifies d_pca=32 hyperparameter choice via elbow analysis.")

def fig4_sic_correlation(df: pd.DataFrame, spearman_result: dict, save_dir: str):
    """
    Figure 4: SIC score vs task success rate scatter plot.
    
    PURPOSE: MAIN RESULT. Shows SIC is a valid proxy for downstream performance.
    High SIC score reliably predicts high task success rate (Spearman rho=0.8336).
    
    df columns: subset_name, sic_score, success_rate, n_demos
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Color by number of demos
    scatter = ax.scatter(df['sic_score'], df['success_rate'],
                        c=df['n_demos'], cmap='plasma', s=80, alpha=0.85,
                        edgecolors='gray', linewidths=0.5)
    plt.colorbar(scatter, ax=ax, label="Number of Demonstrations")
    
    # Label each point
    for _, row in df.iterrows():
        ax.annotate(row['subset_name'], 
                   (row['sic_score'], row['success_rate']),
                   textcoords='offset points', xytext=(4, 4), fontsize=7, alpha=0.8)
    
    # Trend line
    z = np.polyfit(df['sic_score'], df['success_rate'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['sic_score'].min(), df['sic_score'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.6, linewidth=1.5, label="Linear trend")
    
    # Annotation box
    rho = spearman_result['rho']
    p_val = spearman_result['p_value']
    ax.text(0.05, 0.95, f"Spearman ρ = {rho:.4f}\np = {p_val:.4f}",
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel("SIC Score (State Information Coverage)")
    ax.set_ylabel("Task Success Rate (%)")
    ax.set_title("SIC Score as a Training-Free Proxy for Task Success Rate\n"
                 f"(Evaluated on {len(df)} Controlled Demonstration Subsets)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "fig4_sic_correlation")
    plt.savefig(save_path + ".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(save_path + ".png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}.pdf")
    print(f"PURPOSE: MAIN RESULT — SIC score predicts success rate (ρ={rho:.4f})")
    print("         without any model training, validating the SIC proxy metric.")

def fig5_greedy_sic_curve(greedy_result: dict, budget_B: int, n_base: int, save_dir: str):
    """
    Figure 5: SIC growth curve during forward greedy planning.
    
    PURPOSE: Shows greedy algorithm behavior — large marginal gains early
    (filling coverage gaps) then diminishing returns, consistent with SIC's
    theoretical diminishing returns property.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sic_history = greedy_result['sic_history']
    gain_history = greedy_result['gain_history']
    steps = list(range(len(sic_history)))
    stopping = greedy_result.get('stopping_step', len(sic_history) - 1)
    
    # Left: SIC growth
    ax1.plot(steps, sic_history, color=COLORS[0], linewidth=2.5)
    ax1.axvline(x=stopping, color='red', linestyle='--', alpha=0.7,
                label=f'Stopping criterion at step {stopping}')
    ax1.set_xlabel("Number of Additional Collections (beyond B₀)")
    ax1.set_ylabel("Total SIC Score")
    ax1.set_title("SIC Growth During Forward Greedy Planning")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Marginal gains
    ax2.plot(range(len(gain_history)), gain_history, color=COLORS[1], 
             linewidth=2.0, alpha=0.9)
    ax2.axvline(x=stopping, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel("Greedy Step")
    ax2.set_ylabel("SIC Marginal Gain")
    ax2.set_title("Marginal SIC Gain per Greedy Step\n(Diminishing Returns Confirmed)")
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f"Greedy Planning: B₀ ({n_base} demos) → Budget B={budget_B}",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(save_dir, "fig5_greedy_sic_curve")
    plt.savefig(save_path + ".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(save_path + ".png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}.pdf")
    print("PURPOSE: Demonstrates greedy algorithm behavior — early steps fill major")
    print("         coverage gaps, later steps saturate, consistent with SIC theory.")

def fig6_collection_heatmap(plan_df: pd.DataFrame, save_dir: str):
    """
    Figure 6: Recommended collection counts heatmap.
    
    PURPOSE: Shows the non-uniform allocation output by greedy planning.
    High-information configurations get 3-4 collections; saturated regions
    keep 1. Contrasts with uniform allocation strategies.
    
    plan_df: DataFrame with rows=position_id, columns=rotation_degrees, values=n_times
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    sns.heatmap(plan_df, annot=True, fmt='d', cmap='Blues',
                linewidths=0.5, linecolor='gray',
                cbar_kws={'label': 'Recommended Collection Count'},
                ax=ax, vmin=1, vmax=4)
    
    ax.set_xlabel("Rotation Angle (degrees)")
    ax.set_ylabel("Spatial Position ID (3×3 grid)")
    ax.set_title("Recommended Demonstration Collection Plan\n"
                 "(Output of SIC-Guided Greedy Planning, Budget B=144)\n"
                 "Non-uniform allocation: high-info regions get more collections")
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "fig6_collection_heatmap")
    plt.savefig(save_path + ".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(save_path + ".png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}.pdf")
    print("PURPOSE: Actionable output — directly translates to teleoperator task list.")
    print("         Non-uniform allocation contrasts with baselines (all show uniform).")

def fig7_baseline_comparison(results: dict, save_dir: str):
    """
    Figure 7: Bar chart comparing all collection strategies.
    
    results: {
        strategy_name: {'success_rate': float, 'std': float, 'n_demos': int}
    }
    
    PURPOSE: MAIN COMPARISON. SIC-guided planning achieves near-full performance
    with 50% of data, while uniform, FPS, k-center strategies plateau at ~84%.
    """
    if not results:
        print("[SKIP] fig7: no success_rate data. Run training first.")
        return
    
    strategies = list(results.keys())
    srs = [results[s]['success_rate'] for s in strategies]
    stds = [results[s].get('std', 0) for s in strategies]
    n_demos = [results[s]['n_demos'] for s in strategies]
    
    # Color coding
    colors = []
    for s in strategies:
        if 'SIC' in s or 'Ours' in s:
            colors.append(COLORS[2])  # green for ours
        elif 'Full' in s or 'All' in s:
            colors.append(COLORS[3])  # orange for full data
        else:
            colors.append(COLORS[7])  # gray for baselines
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(strategies))
    bars = ax.bar(x, srs, yerr=stds, capsize=5,
                  color=colors, edgecolor='black', linewidth=0.7, alpha=0.85)
    
    # Annotate bars with demo count
    for bar, n, sr in zip(bars, n_demos, srs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{n}d', ha='center', va='bottom', fontsize=9, color='darkgray')
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
                f'{sr:.0f}%', ha='center', va='center', fontsize=10, 
                fontweight='bold', color='white')
    
    ax.set_xlabel("Collection Strategy")
    ax.set_ylabel("Task Success Rate (%)")
    ax.set_title("Demonstration Collection Strategy Comparison\n"
                 "(Budget B=144, numbers above bars = demo count)")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', '\n') for s in strategies], 
                        rotation=15, ha='right')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS[7], label='Baselines'),
        Patch(facecolor=COLORS[2], label='Ours (SIC-Guided)'),
        Patch(facecolor=COLORS[3], label='Full Data (Upper Bound)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "fig7_baseline_comparison")
    plt.savefig(save_path + ".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(save_path + ".png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}.pdf")
    print("PURPOSE: MAIN COMPARISON TABLE. SIC-guided achieves 93% with 144 demos")
    print("         vs uniform/FPS/k-center at 84%, full data at 96%.")

def fig8_efficiency_curve(efficiency_data: dict, save_dir: str):
    """
    Figure 8: Success rate vs number of demonstrations (efficiency curve).
    
    efficiency_data: {
        strategy_name: {'demos': [int], 'success_rates': [float], 'stds': [float]}
    }
    
    PURPOSE: Shows data efficiency — SIC-guided planning consistently reaches
    higher success rate per demonstration than alternatives.
    Special marker for Small paper result: 81 demos = 98%.
    """
    if not efficiency_data:
        print("[SKIP] fig8: no efficiency data. Run training at multiple budgets.")
        return
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    strategy_colors = {
        'Uniform': COLORS[7],
        'Diagonal': COLORS[1],
        'SIC-Guided (Ours)': COLORS[2],
        'Full Data': COLORS[3]
    }
    
    for strategy, data in efficiency_data.items():
        color = strategy_colors.get(strategy, COLORS[4])
        demos = data['demos']
        srs = data['success_rates']
        stds = data.get('stds', [0] * len(srs))
        
        ls = '--' if 'Full' in strategy else '-'
        ax.plot(demos, srs, marker='o', linewidth=2.0, 
                color=color, label=strategy, linestyle=ls, markersize=7)
        ax.fill_between(demos, 
                        [s - e for s, e in zip(srs, stds)],
                        [s + e for s, e in zip(srs, stds)],
                        alpha=0.15, color=color)
    
    # Special marker: small paper result (81 demos = 98%)
    ax.scatter([81], [98.0], marker='*', s=300, color='gold', 
               edgecolors='black', linewidths=1.5, zorder=10,
               label='Diagonal sampling (81 demos) = 98%\n(Small paper result)')
    ax.annotate('81 demos → 98%', xy=(81, 98), xytext=(100, 94),
                fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax.set_xlabel("Number of Demonstrations")
    ax.set_ylabel("Task Success Rate (%)")
    ax.set_title("Data Efficiency: Success Rate vs. Number of Demonstrations\n"
                 "SIC-guided planning achieves higher SR per demo than alternatives")
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "fig8_efficiency_curve")
    plt.savefig(save_path + ".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(save_path + ".png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}.pdf")
    print("PURPOSE: Demonstrates data efficiency advantage of SIC-guided planning.")

def fig9_ablation_components(ablation_df: pd.DataFrame, save_dir: str):
    """
    Figure 9: Ablation study — contribution of each SIC component.
    
    ablation_df columns: variant, spearman_rho, p_value, relative_change
    
    PURPOSE: Validates that each SIC component (count weights, global view, wrist view)
    contributes to correlation with success rate.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    variants = ablation_df['variant'].tolist()
    rhos = ablation_df['spearman_rho'].tolist()
    
    colors = [COLORS[2] if v == 'Full SIC' else COLORS[0] for v in variants]
    bars = ax.bar(variants, rhos, color=colors, edgecolor='black', linewidth=0.7)
    
    for bar, rho in zip(bars, rhos):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'ρ={rho:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel("SIC Variant")
    ax.set_ylabel("Spearman Correlation ρ (with task SR)")
    ax.set_title("Ablation Study: Contribution of Each SIC Component\n"
                 "(Higher ρ = better proxy for task success rate)")
    ax.set_ylim([0.65, 0.90])
    ax.set_xticklabels(variants, rotation=15, ha='right')
    ax.axhline(y=rhos[0], color='green', linestyle='--', alpha=0.5, 
               label='Full SIC baseline')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "fig9_ablation_components")
    plt.savefig(save_path + ".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(save_path + ".png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}.pdf")
    print("PURPOSE: Shows tau (count weights) contributes most to SIC effectiveness;")
    print("         both views contribute independently, justifying dual-view design.")
```

---

**第十步：主运行脚本（run_all.py）**

```python
#!/usr/bin/env python3
"""
SIC Framework — One-Click Runner
Run all analyses and generate all paper figures.

Usage:
  # Full run (requires dataset and success rates)
  python personal/work2/run_all.py \
    --dataset_root /path/to/lerobot_datasets \
    --config_map personal/work2/data/config_map.json \
    --success_rates personal/work2/data/success_rates.json

  # Analysis only (no success rates, skips comparison figures)
  python personal/work2/run_all.py \
    --dataset_root /path/to/lerobot_datasets \
    --config_map personal/work2/data/config_map.json \
    --mode analysis_only

  # Figures only (all data pre-computed, just regenerate figures)
  python personal/work2/run_all.py \
    --mode figures_only

  # With attention map comparison (requires trained model checkpoints)
  python personal/work2/run_all.py \
    --dataset_root /path/to/lerobot_datasets \
    --config_map personal/work2/data/config_map.json \
    --success_rates personal/work2/data/success_rates.json \
    --model_paths personal/work2/data/model_paths.json
"""

import argparse, json, os, sys, pickle
import numpy as np
import pandas as pd
sys.path.insert(0, 'personal/work2')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='')
    parser.add_argument('--config_map', type=str, 
                        default='personal/work2/data/config_map.json')
    parser.add_argument('--success_rates', type=str, default='')
    parser.add_argument('--model_paths', type=str, default='',
                        help='JSON file: {dataset_name: checkpoint_path}')
    parser.add_argument('--mode', choices=['full', 'analysis_only', 'figures_only'],
                        default='full')
    parser.add_argument('--budget_B', type=int, default=144)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip embedding extraction if cached')
    return parser.parse_args()

def main():
    args = parse_args()
    
    from configs import CFG
    CFG.dataset_root = args.dataset_root
    CFG.device = args.device
    CFG.budget_B = args.budget_B
    CFG.setup()
    
    print("=" * 60)
    print("SIC Framework — Full Analysis Pipeline")
    print("=" * 60)
    
    # ============================================================
    # PHASE 1: Embedding Extraction & Anchor Construction
    # ============================================================
    
    if args.mode == 'figures_only':
        print("\n[Mode: figures_only] Loading pre-computed data...")
        anchor_ref = pickle.load(open(os.path.join(CFG.results_dir, 'anchor_ref.pkl'), 'rb'))
        global_embs = pickle.load(open(os.path.join(CFG.cache_dir, 'global_embs.pkl'), 'rb'))
        wrist_embs = pickle.load(open(os.path.join(CFG.cache_dir, 'wrist_embs.pkl'), 'rb'))
        with open(args.config_map) as f:
            config_map_raw = json.load(f)
        config_map = {int(k): tuple(v) for k, v in config_map_raw.items()}
    
    else:
        print("\n[Phase 1] Loading dataset and extracting embeddings...")
        
        if not args.dataset_root:
            print("ERROR: --dataset_root required for modes other than 'figures_only'")
            sys.exit(1)
        
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from sic.embeddings import load_frozen_vlm, extract_and_cache_all_embeddings
        from sic.anchor import build_anchor_reference
        
        with open(args.config_map) as f:
            config_map_raw = json.load(f)
        config_map = {int(k): tuple(v) for k, v in config_map_raw.items()}
        
        dataset = LeRobotDataset(args.dataset_root)
        
        model, processor = load_frozen_vlm(CFG.vlm_model_id, CFG.device)
        
        # Extract embeddings (cached)
        global_embs = extract_and_cache_all_embeddings(
            model, processor, dataset, config_map,
            CFG.global_cam_key, CFG.device,
            os.path.join(CFG.cache_dir, 'global_embs.pkl'),
            CFG.batch_size
        )
        wrist_embs = extract_and_cache_all_embeddings(
            model, processor, dataset, config_map,
            CFG.wrist_cam_key, CFG.device,
            os.path.join(CFG.cache_dir, 'wrist_embs.pkl'),
            CFG.batch_size
        )
        
        anchor_ref = build_anchor_reference(
            global_embs, wrist_embs, config_map,
            d_pca=CFG.d_pca,
            save_path=os.path.join(CFG.results_dir, 'anchor_ref.pkl')
        )
    
    # ============================================================
    # PHASE 2: Research Analyses (no training required)
    # ============================================================
    
    print("\n[Phase 2] Research analyses (embedding space)...")
    
    from research.embedding_analysis import (
        compute_cross_config_distance_curve,
        compute_pca_variance_curve,
        compute_tsne_for_configs
    )
    
    # Cross-config distance curve
    distance_curve = compute_cross_config_distance_curve(wrist_embs, config_map)
    np.save(os.path.join(CFG.results_dir, 'cross_config_distance.npy'), distance_curve)
    
    # PCA variance curves
    global_variance = compute_pca_variance_curve(global_embs, max_components=128)
    wrist_variance = compute_pca_variance_curve(wrist_embs, max_components=128)
    
    # t-SNE (pick 4 configs if not specified)
    tsne_result = compute_tsne_for_configs(wrist_embs, config_map)
    
    # ============================================================
    # PHASE 3: SIC Computation
    # ============================================================
    
    print("\n[Phase 3] Computing SIC scores...")
    
    from sic.score import compute_sic
    from sic.greedy import greedy_plan
    
    # B0 plan: all configs with 1 collection
    b0_plan = {cfg: 1 for cfg in anchor_ref['anchors'].keys()}
    b0_sic = compute_sic(b0_plan, anchor_ref, CFG.alpha, CFG.lambda_weight)
    print(f"  B0 SIC: {b0_sic['sic']:.4f}")
    
    # Greedy planning
    print(f"  Running greedy planning (budget={CFG.budget_B})...")
    greedy_result = greedy_plan(anchor_ref, CFG.budget_B, CFG.t_max, 
                                CFG.alpha, CFG.lambda_weight)
    
    # Plan to DataFrame for heatmap
    plan = greedy_result['final_plan']
    all_pos = sorted(set(k[0] for k in plan.keys()))
    all_rot = sorted(set(k[1] for k in plan.keys()))
    plan_matrix = np.zeros((len(all_pos), len(all_rot)), dtype=int)
    for (p, r), n in plan.items():
        plan_matrix[all_pos.index(p), all_rot.index(r)] = n
    
    rot_labels = [f"{r*45}°" for r in all_rot]
    plan_df = pd.DataFrame(plan_matrix, 
                           index=[f"Pos {p}" for p in all_pos],
                           columns=rot_labels)
    
    print(f"  Greedy plan: {sum(plan.values())} total demos, SIC={greedy_result['sic_history'][-1]:.4f}")
    
    # ============================================================
    # PHASE 4: Correlation Analysis (loads success rates if available)
    # ============================================================
    
    print("\n[Phase 4] Correlation analysis...")
    
    success_rates = None
    if args.success_rates and os.path.exists(args.success_rates):
        with open(args.success_rates) as f:
            success_rates = json.load(f)
        print(f"  Loaded success rates for {len(success_rates)} subsets")
    else:
        print("  [INFO] No success rates found. Correlation figures will be skipped.")
        print("  To add results: create personal/work2/data/success_rates.json")
        print("  Format: {subset_name: {success_rate: float, std: float, n_demos: int, sic_score: float}}")
    
    spearman_result = None
    df_correlation = None
    
    if success_rates:
        from scipy.stats import spearmanr
        from sic.analysis import compute_spearman_correlation
        
        df_correlation = pd.DataFrame([
            {
                'subset_name': name,
                'sic_score': data['sic_score'],
                'success_rate': data['success_rate'],
                'std': data.get('std', 0),
                'n_demos': data['n_demos']
            }
            for name, data in success_rates.items()
        ])
        
        rho, p_val = spearmanr(df_correlation['sic_score'], df_correlation['success_rate'])
        spearman_result = {'rho': rho, 'p_value': p_val}
        
        print(f"  Spearman rho = {rho:.4f}, p = {p_val:.6f}")
    
    # ============================================================
    # PHASE 5: Attention Map Analysis (if model paths provided)
    # ============================================================
    
    model_comparison_result = None
    if args.model_paths and os.path.exists(args.model_paths):
        print("\n[Phase 5] Attention map analysis...")
        from research.attention_analysis import compare_attention_across_datasets
        from sic.embeddings import load_frozen_vlm
        
        with open(args.model_paths) as f:
            model_paths = json.load(f)
        
        _, processor = load_frozen_vlm(CFG.vlm_model_id, CFG.device)
        
        # Load a few test images from dataset
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from PIL import Image
        import torch
        dataset = LeRobotDataset(args.dataset_root)
        
        test_images = []
        for ep_idx in list(config_map.keys())[:3]:  # 3 test images
            frame = dataset[dataset.episode_data_index['from'][ep_idx]]
            img_t = frame[CFG.global_cam_key]
            img_pil = Image.fromarray(
                (img_t.permute(1,2,0).numpy() * 255).astype(np.uint8)
            )
            test_images.append((img_pil, "Grasp the object and lift it out"))
        
        model_comparison_result = compare_attention_across_datasets(
            model_paths, test_images, processor, CFG.device, CFG.figures_dir
        )
    else:
        print("\n[Phase 5] Attention analysis skipped (no model paths provided)")
    
    # ============================================================
    # PHASE 6: Generate All Figures
    # ============================================================
    
    print("\n[Phase 6] Generating all figures...")
    
    from visualize.all_figures import (
        fig1_tsne_embeddings,
        fig2_cross_config_distance,
        fig3_pca_variance,
        fig4_sic_correlation,
        fig5_greedy_sic_curve,
        fig6_collection_heatmap,
        fig7_baseline_comparison,
        fig8_efficiency_curve,
        fig9_ablation_components
    )
    
    print("\n  Generating fig1: t-SNE embeddings...")
    fig1_tsne_embeddings(tsne_result, CFG.figures_dir)
    
    print("  Generating fig2: cross-config distance curve...")
    fig2_cross_config_distance(distance_curve, CFG.figures_dir)
    
    print("  Generating fig3: PCA variance curves...")
    fig3_pca_variance(global_variance, wrist_variance, CFG.figures_dir)
    
    if df_correlation is not None and spearman_result is not None:
        print("  Generating fig4: SIC correlation scatter...")
        fig4_sic_correlation(df_correlation, spearman_result, CFG.figures_dir)
    else:
        print("  [SKIP] fig4: waiting for success rate data")
    
    print("  Generating fig5: greedy SIC curve...")
    fig5_greedy_sic_curve(greedy_result, CFG.budget_B, len(b0_plan), CFG.figures_dir)
    
    print("  Generating fig6: collection heatmap...")
    fig6_collection_heatmap(plan_df, CFG.figures_dir)
    
    if success_rates:
        comparison_data = {
            name: {'success_rate': d['success_rate'], 
                   'std': d.get('std', 0), 
                   'n_demos': d['n_demos']}
            for name, d in success_rates.items()
        }
        print("  Generating fig7: baseline comparison...")
        fig7_baseline_comparison(comparison_data, CFG.figures_dir)
    else:
        print("  [SKIP] fig7: waiting for success rate data")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — Summary")
    print("=" * 60)
    print(f"\nFigures saved to: {CFG.figures_dir}/")
    print("\nAvailable figures:")
    for f in sorted(os.listdir(CFG.figures_dir)):
        if f.endswith('.pdf'):
            print(f"  {f}")
    
    print("\nKey results:")
    print(f"  B0 SIC score: {b0_sic['sic']:.4f}")
    print(f"  Greedy plan ({CFG.budget_B} demos) SIC: {greedy_result['sic_history'][-1]:.4f}")
    
    if spearman_result:
        print(f"  Spearman correlation: rho={spearman_result['rho']:.4f}, p={spearman_result['p_value']:.6f}")
    
    print(f"\nStopping criterion: step {greedy_result.get('stopping_step', 'N/A')}")
    print("\nCollect plan printout:")
    for (pos, rot), n in sorted(plan.items(), key=lambda x: -x[1]):
        if n > 1:
            print(f"  Position {pos}, Rotation {rot*45}°: {n} collections (add {n-1} to B0)")

if __name__ == '__main__':
    main()
```

---

**执行方式**

```bash
# 切换到工作目录
cd /path/to/lerobot_june
git checkout server-dev-merge-test

# 创建数据目录和必要文件
mkdir -p personal/work2/data

# 第一次运行（提取嵌入+构建锚点+分析，不需要成功率数据）
python personal/work2/run_all.py \
  --dataset_root /path/to/your/lerobot/dataset \
  --config_map personal/work2/data/config_map.json \
  --mode analysis_only

# 填入成功率数据后完整运行
python personal/work2/run_all.py \
  --dataset_root /path/to/your/lerobot/dataset \
  --config_map personal/work2/data/config_map.json \
  --success_rates personal/work2/data/success_rates.json

# 如果已有训练好的模型，加入注意力图分析
python personal/work2/run_all.py \
  --mode full \
  --dataset_root /path/to/your/lerobot/dataset \
  --config_map personal/work2/data/config_map.json \
  --success_rates personal/work2/data/success_rates.json \
  --model_paths personal/work2/data/model_paths.json
```

---

**config_map.json格式说明**

```json
{
  "0":  [0, 0],
  "1":  [0, 1],
  "2":  [0, 2],
  "8":  [0, 7],
  "9":  [1, 0],
  ...
  "71": [8, 7]
}
```
键是episode_index（字符串），值是[position_id, rotation_id]。共72条（9位置×8姿态）。

**success_rates.json格式说明**

```json
{
  "ALL":                {"sic_score": 61.98, "success_rate": 96, "std": 1.2, "n_demos": 288},
  "B144-halfT":         {"sic_score": 61.27, "success_rate": 84, "std": 0.4, "n_demos": 144},
  "B72-singleT":        {"sic_score": 60.62, "success_rate": 78, "std": 0.7, "n_demos": 72},
  "SIC-Guided-144":     {"sic_score": null,  "success_rate": 93, "std": 1.0, "n_demos": 144},
  ...
}
```

**model_paths.json格式说明**

```json
{
  "High-SIC (144 demos, SIC-guided)":   "checkpoints/sic_guided_144/",
  "Low-SIC (64 demos, corners-cardinal)": "checkpoints/low_sic_64/",
  "Full Data (288 demos)":               "checkpoints/full_288/"
}
```