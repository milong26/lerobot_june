"""
代码的功能：
提取root=/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/uniform_pickplacev3
的数据集，仿照personal/work2/dataset_lookin/see_uniform_dataset.py的功能，找到两个x和y相差很大的episode。
将episode的内容按照
src/lerobot/policies/smolvla/modeling_smolvla.py
src/lerobot/policies/smolvla/configuration_smolvla.py
src/lerobot/policies/smolvla/processor_smolvla.py
src/lerobot/policies/smolvla/smolvlm_with_expert.py
的方式，使用vlm模型HuggingFaceTB/SmolVLM2-500M-Video-Instruct处理后比较这两集的embedding
"""
"""
用什么方式分析比较好？
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lerobot.datasets import LeRobotDataset

# ─── 配置 ─────────────────────────────────────────────────────────────
DATASET_ROOT = "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pickplacev3"
EPISODE_STATES_PATH = "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pickplacev3/episode_initial_states.json"
# DATASET_ROOT = "/data/zhonglinye/jun/lerobot/personal/work2/dataset"

# EPISODE_STATES_PATH = "/data/zhonglinye/jun/lerobot/personal/work2/dataset/episode_initial_states.json"
MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
TASK_PROMPT = "pick and place"
OUTPUT_DIR = str(Path(__file__).parent / "results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

EMBEDDING_DIR = str(Path(__file__).parent)


def save_embeddings(emb_a, emb_b, ep_a, ep_b):
    """保存embedding到文件"""
    path_a = f"{EMBEDDING_DIR}/embedding_ep{ep_a}.pt"
    path_b = f"{EMBEDDING_DIR}/embedding_ep{ep_b}.pt"
    torch.save(emb_a, path_a)
    torch.save(emb_b, path_b)
    print(f"Embedding已保存: {path_a}, {path_b}")


def load_embeddings(ep_a, ep_b):
    """从文件加载embedding"""
    path_a = f"{EMBEDDING_DIR}/embedding_ep{ep_a}.pt"
    path_b = f"{EMBEDDING_DIR}/embedding_ep{ep_b}.pt"
    if Path(path_a).exists() and Path(path_b).exists():
        emb_a = torch.load(path_a, weights_only=True)
        emb_b = torch.load(path_b, weights_only=True)
        print(f"从缓存加载Embedding: {path_a}, {path_b}")
        return emb_a, emb_b
    return None, None


def find_most_different_episodes():
    """找到两个x和y相差最大的episode"""
    print("加载episode初始状态...")
    with open(EPISODE_STATES_PATH, "r") as f:
        states_data = json.load(f)

    episodes = states_data["episodes"]
    print(f"总episode数: {len(episodes)}")

    max_dist = 0
    best_pair = (0, 1)

    # 采样比较，避免O(n^2)全量计算
    step = max(1, len(episodes) // 100)
    candidates = episodes[::step]

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            xi, yi = candidates[i]["obj_init_pos"][0], candidates[i]["obj_init_pos"][1]
            xj, yj = candidates[j]["obj_init_pos"][0], candidates[j]["obj_init_pos"][1]
            dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            if dist > max_dist:
                max_dist = dist
                best_pair = (candidates[i]["episode_index"], candidates[j]["episode_index"])

    ep_a, ep_b = best_pair
    pos_a = episodes[ep_a]["obj_init_pos"][:2]
    pos_b = episodes[ep_b]["obj_init_pos"][:2]

    print(f"找到差异最大的episode对:")
    print(f"  Episode {ep_a}: obj_init_pos=({pos_a[0]:.4f}, {pos_a[1]:.4f})")
    print(f"  Episode {ep_b}: obj_init_pos=({pos_b[0]:.4f}, {pos_b[1]:.4f})")
    print(f"  XY距离: {max_dist:.4f}")

    return ep_a, ep_b


def load_episode_frames(dataset, ep_idx):
    """加载episode的所有帧"""
    from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
    to_idx = dataset.meta.episodes["dataset_to_index"][ep_idx]

    frames = []
    for idx in range(int(from_idx), int(to_idx)):
        frame = dataset[int(idx)]
        img = frame["observation.images.top"]  # (C, H, W)
        frames.append(img)

    return torch.stack(frames)


def extract_embeddings(frames, model, processor, batch_size=8):
    """提取所有帧的embedding"""
    embeddings = []

    # 获取image token的占位符
    image_token = processor.tokenizer.image_token if hasattr(processor.tokenizer, 'image_token') else "<image>"

    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_size_actual = len(batch_frames)

            # 逐帧处理：每张图对应一个文本
            batch_embeddings = []
            for frame in batch_frames:
                text = f"{image_token}\n{TASK_PROMPT}"
                inputs = processor(
                    text=text,
                    images=[frame],
                    return_tensors="pt",
                ).to(DEVICE)

                outputs = model(**inputs, output_hidden_states=True)

                last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
                attention_mask = inputs.get("attention_mask", None)

                if attention_mask is not None:
                    mask = attention_mask.unsqueeze(-1).float()
                    mean_emb = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                else:
                    mean_emb = last_hidden.mean(dim=1)

                batch_embeddings.append(mean_emb.cpu().squeeze(0))

            embeddings.extend(batch_embeddings)

            if (i // batch_size) % 10 == 0:
                print(f"  处理进度: {i}/{len(frames)} 帧")

    return torch.stack(embeddings, dim=0)


def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def euclidean_distance(a, b):
    """计算两个向量的欧氏距离"""
    return torch.norm(a - b, dim=-1)


# ─── 方案1: 逐帧余弦相似度曲线 ───────────────────────────────────────
def analyze_cosine_similarity(emb_a, emb_b, ep_a, ep_b):
    """方案1: 计算对应时间步的余弦相似度和欧氏距离"""
    print("\n" + "=" * 60)
    print("方案1: 逐帧余弦相似度 & 欧氏距离分析")
    print("=" * 60)

    min_len = min(len(emb_a), len(emb_b))
    emb_a_trim = emb_a[:min_len]
    emb_b_trim = emb_b[:min_len]

    cos_sims = cosine_similarity(emb_a_trim, emb_b_trim)
    eucl_dists = euclidean_distance(emb_a_trim, emb_b_trim)

    time_steps = np.arange(min_len)

    # 分段统计
    n = min_len
    early_end = n // 3
    late_start = 2 * n // 3

    early_sim = cos_sims[:early_end].mean().item()
    mid_sim = cos_sims[early_end:late_start].mean().item()
    late_sim = cos_sims[late_start:].mean().item()

    print(f"  总帧数: {min_len}")
    print(f"  初期(0-33%)平均余弦相似度: {early_sim:.4f}")
    print(f"  中期(33-66%)平均余弦相似度: {mid_sim:.4f}")
    print(f"  后期(66-100%)平均余弦相似度: {late_sim:.4f}")
    print(f"  初期→后期变化: {late_sim - early_sim:+.4f}")

    # 可区分度指标
    early_dist = eucl_dists[:early_end].mean().item()
    late_dist = eucl_dists[late_start:].mean().item()
    if late_dist > 0:
        distinguishability = early_dist / late_dist
        print(f"  可区分度(初期距离/后期距离): {distinguishability:.2f}")
        if distinguishability > 2:
            print("  ✅ 模型能很好地区分初期和后期")
        elif distinguishability > 1.2:
            print("  ⚠️  模型有一定区分能力，但不够明显")
        else:
            print("  ❌ 模型对位置变化不敏感")

    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(time_steps, cos_sims.numpy(), linewidth=2, color="steelblue")
    ax1.axvline(x=early_end, color="gray", linestyle="--", alpha=0.5)
    ax1.axvline(x=late_start, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Time Step (Frame Index)", fontsize=12)
    ax1.set_ylabel("Cosine Similarity", fontsize=12)
    ax1.set_title(f"Episode {ep_a} vs Episode {ep_b}: Frame-wise Cosine Similarity", fontsize=14)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.text(early_end / 2, 0.95, "Early", ha="center", fontsize=10, color="gray")
    ax1.text((early_end + late_start) / 2, 0.95, "Mid", ha="center", fontsize=10, color="gray")
    ax1.text(late_start + (min_len - late_start) / 2, 0.95, "Late", ha="center", fontsize=10, color="gray")

    ax2.plot(time_steps, eucl_dists.numpy(), linewidth=2, color="coral")
    ax2.axvline(x=early_end, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(x=late_start, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Time Step (Frame Index)", fontsize=12)
    ax2.set_ylabel("Euclidean Distance", fontsize=12)
    ax2.set_title(f"Episode {ep_a} vs Episode {ep_b}: Frame-wise Euclidean Distance", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/01_cosine_similarity.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  图表已保存: {out_path}")

    return cos_sims, eucl_dists


# ─── 方案2: PCA降维可视化 ───────────────────────────────────────────
def analyze_pca_visualization(emb_a, emb_b, ep_a, ep_b):
    """方案2: PCA降维到2D，可视化embedding轨迹"""
    print("\n" + "=" * 60)
    print("方案2: PCA降维可视化")
    print("=" * 60)

    all_embs = torch.cat([emb_a, emb_b], dim=0).numpy()
    n_a = len(emb_a)
    n_b = len(emb_b)

    # PCA降维
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(all_embs)

    print(f"  PCA解释方差比: {pca.explained_variance_ratio_}")
    print(f"  总解释方差: {pca.explained_variance_ratio_.sum():.2%}")

    emb_a_2d = emb_2d[:n_a]
    emb_b_2d = emb_2d[n_a:]

    fig, ax = plt.subplots(figsize=(10, 8))

    # 用颜色深浅表示时间
    cmap_a = plt.cm.Blues
    cmap_b = plt.cm.Reds

    time_a = np.linspace(0, 1, n_a)
    time_b = np.linspace(0, 1, n_b)

    scatter_a = ax.scatter(emb_a_2d[:, 0], emb_a_2d[:, 1], c=time_a, cmap=cmap_a,
                           s=30, alpha=0.7, edgecolors="navy", linewidth=0.3, label=f"Episode {ep_a}")
    scatter_b = ax.scatter(emb_b_2d[:, 0], emb_b_2d[:, 1], c=time_b, cmap=cmap_b,
                           s=30, alpha=0.7, edgecolors="darkred", linewidth=0.3, label=f"Episode {ep_b}")

    # 画轨迹线
    ax.plot(emb_a_2d[:, 0], emb_a_2d[:, 1], color="steelblue", alpha=0.3, linewidth=1)
    ax.plot(emb_b_2d[:, 0], emb_b_2d[:, 1], color="salmon", alpha=0.3, linewidth=1)

    # 标记起点和终点
    ax.scatter(emb_a_2d[0, 0], emb_a_2d[0, 1], c="blue", s=100, marker="^", label=f"Ep{ep_a} Start")
    ax.scatter(emb_a_2d[-1, 0], emb_a_2d[-1, 1], c="blue", s=100, marker="s", label=f"Ep{ep_a} End")
    ax.scatter(emb_b_2d[0, 0], emb_b_2d[0, 1], c="red", s=100, marker="^", label=f"Ep{ep_b} Start")
    ax.scatter(emb_b_2d[-1, 0], emb_b_2d[-1, 1], c="red", s=100, marker="s", label=f"Ep{ep_b} End")

    ax.set_xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=12)
    ax.set_ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=12)
    ax.set_title(f"PCA: Episode {ep_a} vs {ep_b} Embedding Trajectory", fontsize=14)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/02_pca_visualization.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  图表已保存: {out_path}")

    # 计算起点和终点的距离
    start_dist = np.linalg.norm(emb_a_2d[0] - emb_b_2d[0])
    end_dist = np.linalg.norm(emb_a_2d[-1] - emb_b_2d[-1])
    print(f"  PCA空间中起点距离: {start_dist:.4f}")
    print(f"  PCA空间中终点距离: {end_dist:.4f}")
    print(f"  起点/终点距离比: {start_dist / end_dist:.2f}")


# ─── 方案3: 相似度矩阵 ──────────────────────────────────────────────
def analyze_similarity_matrix(emb_a, emb_b, ep_a, ep_b):
    """方案3: 交叉时间步相似度矩阵"""
    print("\n" + "=" * 60)
    print("方案3: 交叉时间步相似度矩阵")
    print("=" * 60)

    n_a = len(emb_a)
    n_b = len(emb_b)

    # 如果帧数太多，降采样
    max_frames = 60
    if n_a > max_frames or n_b > max_frames:
        idx_a = np.linspace(0, n_a - 1, max_frames, dtype=int)
        idx_b = np.linspace(0, n_b - 1, max_frames, dtype=int)
        emb_a_sampled = emb_a[idx_a]
        emb_b_sampled = emb_b[idx_b]
        print(f"  降采样: {n_a}→{max_frames}, {n_b}→{max_frames}")
    else:
        emb_a_sampled = emb_a
        emb_b_sampled = emb_b

    # 计算相似度矩阵
    sim_matrix = torch.mm(
        torch.nn.functional.normalize(emb_a_sampled, dim=-1),
        torch.nn.functional.normalize(emb_b_sampled, dim=-1).T
    ).numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_matrix, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel(f"Episode {ep_b} Time Step", fontsize=12)
    ax.set_ylabel(f"Episode {ep_a} Time Step", fontsize=12)
    ax.set_title(f"Cross-temporal Cosine Similarity Matrix\nEp{ep_a} vs Ep{ep_b}", fontsize=14)
    plt.colorbar(im, ax=ax, label="Cosine Similarity")

    # 画对角线
    min_dim = min(sim_matrix.shape)
    ax.plot(range(min_dim), range(min_dim), "r--", alpha=0.5, linewidth=2, label="Diagonal")
    ax.legend()

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/03_similarity_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  图表已保存: {out_path}")

    # 分析对角线强度
    diag_values = np.diag(sim_matrix)
    off_diag_1 = np.diag(sim_matrix, k=1)
    off_diag_5 = np.diag(sim_matrix, k=5) if sim_matrix.shape[0] > 5 else np.array([])

    print(f"  Diagonal avg similarity: {diag_values.mean():.4f}")
    print(f"  Offset-1 avg similarity: {off_diag_1.mean():.4f}")
    if len(off_diag_5) > 0:
        print(f"  Offset-5 avg similarity: {off_diag_5.mean():.4f}")


# ─── 方案4: 可区分度分类实验 ────────────────────────────────────────
def analyze_classification(emb_a, emb_b, ep_a, ep_b):
    """方案4: 用简单分类器测试可区分度"""
    print("\n" + "=" * 60)
    print("方案4: 分类可区分度分析")
    print("=" * 60)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    n_a = len(emb_a)
    n_b = len(emb_b)
    min_len = min(n_a, n_b)

    # 分初期、中期、后期
    early_end = min_len // 3
    late_start = 2 * min_len // 3

    phases = [
        ("Early", 0, early_end),
        ("Mid", early_end, late_start),
        ("Late", late_start, min_len),
    ]

    for phase_name, start, end in phases:
        X_a = emb_a[start:end].numpy()
        X_b = emb_b[start:end].numpy()

        X = np.vstack([X_a, X_b])
        y = np.array([0] * len(X_a) + [1] * len(X_b))

        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, X, y, cv=min(5, len(X_a), len(X_b)), scoring="accuracy")

        print(f"  {phase_name}: Accuracy = {scores.mean():.2%} (+-{scores.std():.2%})")

        if scores.mean() > 0.8:
            print(f"    Model can well distinguish the two episodes")
        elif scores.mean() > 0.6:
            print(f"    Model has some distinguishing ability")
        else:
            print(f"    Model struggles to distinguish the two episodes")

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 5))
    phase_names = [p[0] for p in phases]
    phase_means = []
    phase_stds = []

    for phase_name, start, end in phases:
        X_a = emb_a[start:end].numpy()
        X_b = emb_b[start:end].numpy()
        X = np.vstack([X_a, X_b])
        y = np.array([0] * len(X_a) + [1] * len(X_b))
        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, X, y, cv=min(5, len(X_a), len(X_b)), scoring="accuracy")
        phase_means.append(scores.mean())
        phase_stds.append(scores.std())

    ax.bar(phase_names, phase_means, yerr=phase_stds, capsize=5,
           color=["steelblue", "goldenrod", "coral"], alpha=0.8, edgecolor="black")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random Baseline (50%)")
    ax.set_ylabel("Classification Accuracy", fontsize=12)
    ax.set_title(f"Episode {ep_a} vs {ep_b}: Phase-wise Distinguishability", fontsize=14)
    ax.set_ylim(0.3, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/04_classification_accuracy.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  图表已保存: {out_path}")


# ─── 方案5: t-SNE可视化 ─────────────────────────────────────────────
def analyze_tsne_visualization(emb_a, emb_b, ep_a, ep_b):
    """方案5: t-SNE降维可视化"""
    print("\n" + "=" * 60)
    print("方案5: t-SNE降维可视化")
    print("=" * 60)

    all_embs = torch.cat([emb_a, emb_b], dim=0).numpy()
    n_a = len(emb_a)
    n_b = len(emb_b)

    # 先用PCA降维到50维加速t-SNE
    pca = PCA(n_components=50, random_state=42)
    all_embs_pca = pca.fit_transform(all_embs)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embs) - 1))
    emb_2d = tsne.fit_transform(all_embs_pca)

    emb_a_2d = emb_2d[:n_a]
    emb_b_2d = emb_2d[n_a:]

    fig, ax = plt.subplots(figsize=(10, 8))

    time_a = np.linspace(0, 1, n_a)
    time_b = np.linspace(0, 1, n_b)

    ax.scatter(emb_a_2d[:, 0], emb_a_2d[:, 1], c=time_a, cmap="Blues",
               s=30, alpha=0.7, edgecolors="navy", linewidth=0.3, label=f"Episode {ep_a}")
    ax.scatter(emb_b_2d[:, 0], emb_b_2d[:, 1], c=time_b, cmap="Reds",
               s=30, alpha=0.7, edgecolors="darkred", linewidth=0.3, label=f"Episode {ep_b}")

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title(f"t-SNE: Episode {ep_a} vs {ep_b} Embedding Distribution", fontsize=14)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/05_tsne_visualization.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  图表已保存: {out_path}")


# ─── 主函数 ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Episode Embedding 比较分析")
    print("=" * 60)

    # 1. 找到差异最大的两个episode
    ep_a, ep_b = find_most_different_episodes()

    # 2. 加载数据集
    print(f"\n加载数据集: {DATASET_ROOT}")
    dataset = LeRobotDataset(
        repo_id="work2/metaworld_pick_place",
        root=DATASET_ROOT
    )
    print(f"总episode数: {dataset.num_episodes}")

    # 3. 加载帧
    print(f"\n加载Episode {ep_a} 的帧...")
    frames_a = load_episode_frames(dataset, ep_a)
    print(f"  加载了 {len(frames_a)} 帧")

    print(f"\n加载Episode {ep_b} 的帧...")
    frames_b = load_episode_frames(dataset, ep_b)
    print(f"  加载了 {len(frames_b)} 帧")

    # 5. 提取或加载embedding
    emb_a, emb_b = load_embeddings(ep_a, ep_b)

    if emb_a is None or emb_b is None:
        # 需要提取
        print(f"\n加载模型: {MODEL_ID}")
        print(f"设备: {DEVICE}")

        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
        )
        if DEVICE == "cpu":
            model = model.to(DEVICE)
        model.eval()

        print(f"\n提取Episode {ep_a} 的embedding...")
        emb_a = extract_embeddings(frames_a, model, processor, batch_size=8)
        print(f"  Embedding形状: {emb_a.shape}")

        print(f"\n提取Episode {ep_b} 的embedding...")
        emb_b = extract_embeddings(frames_b, model, processor, batch_size=8)
        print(f"  Embedding形状: {emb_b.shape}")

        # 保存embedding
        save_embeddings(emb_a, emb_b, ep_a, ep_b)
    else:
        print(f"\n使用缓存的Embedding")
        print(f"  Episode {ep_a} Embedding形状: {emb_a.shape}")
        print(f"  Episode {ep_b} Embedding形状: {emb_b.shape}")

    # 6. 执行多种分析方案
    print("\n" + "=" * 60)
    print("开始分析...")
    print("=" * 60)

    # 方案1: 余弦相似度曲线
    cos_sims, eucl_dists = analyze_cosine_similarity(emb_a, emb_b, ep_a, ep_b)

    # 方案2: PCA可视化
    analyze_pca_visualization(emb_a, emb_b, ep_a, ep_b)

    # 方案3: 相似度矩阵
    analyze_similarity_matrix(emb_a, emb_b, ep_a, ep_b)

    # 方案4: 分类可区分度
    analyze_classification(emb_a, emb_b, ep_a, ep_b)

    # 方案5: t-SNE可视化
    analyze_tsne_visualization(emb_a, emb_b, ep_a, ep_b)

    print("\n" + "=" * 60)
    print("分析完成!")
    print(f"所有结果已保存到: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()