# -*- coding: utf-8 -*-
"""
SIC-Real 分析脚本
=================
核心改动：每条 trial 视为独立个体（去掉 τ 折扣），
直接用 PCA 嵌入计算覆盖度，更贴近真实采集差异。

公式：
  SIC(S) = Σ_a f(Σ_{i∈S} K_side(a,i))
          + λ · Σ_a f(Σ_{i∈S} K_wrist(a,i))
  f(x) = x / (1 + x)  （饱和函数，自然实现边际递减）

分析内容：
  1. 相关性分析（散点图）
  2. 留一法相关性分析（LOO）
  3. 消融实验
  4. 超参数分析（lambda 热力图/曲线）
  5. 噪声估计（同 (p,r) 下多 trial 的嵌入方差）
  6. 新旧 SIC 对比
"""

import os
import pickle
import warnings
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════
# 路径与常量
# ═══════════════════════════════════════════════════════
CACHE_DIR = "experiment_results_v5/cache"
SAVE_DIR  = "419_result_but_real"
os.makedirs(SAVE_DIR, exist_ok=True)

D_PCA = 64

# 最佳基础配置（继承自 sweep 结果）
BEST_DBAR   = "mean"
BEST_CPLX   = "path_length"
BEST_KERNEL = "rbf"
BEST_LAMBDA = 1.0   # 新框架无 α，只有 λ

# n=12 显著性参考线（用于相关性图标注）
SIG_LEVELS = {
    "p<0.10": 0.503,
    "p<0.05": 0.587,
    "p<0.01": 0.727,
}

# ═══════════════════════════════════════════════════════
# 中文字体
# ═══════════════════════════════════════════════════════
def setup_chinese_font(
    font_path="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
):
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"找不到中文字体: {font_path}")
    font_manager.fontManager.addfont(font_path)
    cn_font = font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams["font.family"] = cn_font.get_name()
    matplotlib.rcParams["axes.unicode_minus"] = False
    return cn_font

def set_cn_labels(xlabel=None, ylabel=None, cn_font=None,
                  xlabel_size=12, ylabel_size=12):
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=xlabel_size, fontproperties=cn_font)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=ylabel_size, fontproperties=cn_font)

def ax_set_cn_labels(ax, xlabel=None, ylabel=None, cn_font=None,
                     xlabel_size=11, ylabel_size=11):
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=xlabel_size, fontproperties=cn_font)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=ylabel_size, fontproperties=cn_font)

# ═══════════════════════════════════════════════════════
# 子集定义与成功率
# ═══════════════════════════════════════════════════════
SUCCESS_RATE = {
    "all":                              0.96,
    "B144_fullPR_halfT":                0.84,
    "B72_fullPR_singleT1":             0.78,
    "P_cross":                          0.85,
    "B144_fullP_halfR_fullT_diagonal":  0.86,
    "B72_fullP_halfR_halfT_diagonal":   0.74,
    "P_corners_and_center":             0.70,
    "B144_fullP_halfR_fullT_cardinal":  0.70,
    "B72_fullP_halfR_halfT_cardinal":   0.60,
    "P_corners":                        0.58,
    "B64_corners_fullR_halfT":          0.45,
    "B64_corners_halfR_fullT_cardinal": 0.48,
}

SUBSET_CN = {
    "all":                              "全集",
    "B144_fullPR_halfT":               "B144全PR半T",
    "B72_fullPR_singleT1":            "B72全PR单T",
    "P_cross":                         "十字位置",
    "B144_fullP_halfR_fullT_diagonal": "B144对角旋转",
    "B72_fullP_halfR_halfT_diagonal":  "B72对角旋转",
    "P_corners_and_center":            "角+中心",
    "B144_fullP_halfR_fullT_cardinal": "B144基本方向",
    "B72_fullP_halfR_halfT_cardinal":  "B72基本方向",
    "P_corners":                       "四角位置",
    "B64_corners_fullR_halfT":         "B64角全旋转",
    "B64_corners_halfR_fullT_cardinal":"B64角基本方向",
}

def get_subset_dict(name: str) -> dict:
    pos_all = list(range(1, 10))
    rot_all = list(range(8))
    p_list = (
        [1, 3, 5, 7, 9] if "corners_and_center" in name else
        [1, 3, 7, 9]     if "corners"            in name else
        [2, 4, 5, 6, 8]  if "cross"              in name else
        pos_all
    )
    r_list = (
        [1, 3, 5, 7] if "diagonal" in name else
        [0, 2, 4, 6] if "cardinal" in name else
        rot_all
    )
    t_val = 1 if "singleT" in name else (2 if "halfT" in name else 4)
    return {(p, r): t_val for p in p_list for r in r_list}

# ═══════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════
def compute_kernel(x, y, d_bar, mode="rbf"):
    if mode == "rbf":
        return float(np.exp(-np.linalg.norm(x - y) / d_bar))
    elif mode == "cosine":
        num   = np.dot(x, y)
        denom = np.linalg.norm(x) * np.linalg.norm(y) + 1e-8
        return float(0.5 * (1.0 + num / denom))
    raise ValueError(f"Unknown kernel: {mode}")

def sat(x):
    """饱和函数 f(x) = x / (1 + x)"""
    return x / (1.0 + x)

def correlate(scores, srs):
    if len(scores) < 3:
        return 0.0, 1.0, 0.0
    rho, p_s = spearmanr(scores, srs)
    tau_k, _ = kendalltau(scores, srs)
    return float(rho), float(p_s), float(tau_k)

# ═══════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════
def load_cache():
    print("[加载] 从缓存读取 trials / PCA / anchors ...")

    with open(os.path.join(CACHE_DIR, "all_trials.pkl"), "rb") as f:
        trials = pickle.load(f)

    with open(os.path.join(CACHE_DIR, f"pca_d{D_PCA}.pkl"), "rb") as f:
        pca_models = pickle.load(f)

    anchor_key = f"anchors_{BEST_DBAR}_{BEST_CPLX}"
    with open(os.path.join(CACHE_DIR, f"{anchor_key}.pkl"), "rb") as f:
        anchors, d_bar_side, d_bar_wrist = pickle.load(f)

    print(f"  trials={len(trials)}, anchors={len(anchors)}")
    print(f"  d_bar_side={d_bar_side:.2f}, d_bar_wrist={d_bar_wrist:.2f}")
    return trials, pca_models, anchors, d_bar_side, d_bar_wrist

# ═══════════════════════════════════════════════════════
# 提取嵌入：每条 trial → 独立一个特征向量
# ═══════════════════════════════════════════════════════
def extract_trial_embedding(td, pca_models):
    """单条 trial → (side_first, wrist_repr)"""
    w_pca = pca_models["wrist"].transform(td["wrist"])
    s_pca = pca_models["side"].transform(td["side"])
    side_first = s_pca[0]
    F = len(w_pca)
    if F >= 5:
        start = max(1, int(F * 0.3))
        end   = min(F, int(F * 0.7))
        wrist_repr = w_pca[start:end].mean(axis=0)
    else:
        wrist_repr = w_pca.mean(axis=0)
    return {"side_first": side_first, "wrist_repr": wrist_repr}

def build_subset_trial_list(name, trials, pca_models):
    """
    将子集定义转换为 trial 嵌入的 **平铺列表**（每条 trial 独立）。
    同时记录每条 trial 对应的 (p,r,t) 便于调试。
    """
    sd = get_subset_dict(name)
    emb_list = []
    meta_list = []
    for (p, r), T in sd.items():
        for t in range(1, T + 1):
            key = (p, r, t)
            if key not in trials:
                continue
            emb = extract_trial_embedding(trials[key], pca_models)
            emb_list.append(emb)
            meta_list.append(key)
    return emb_list, meta_list

def build_all_subset_embs(trials, pca_models):
    print("[提取] 所有子集的 trial 嵌入（平铺列表）...")
    subset_embs = {}
    subset_meta = {}
    for name in SUCCESS_RATE:
        embs, metas = build_subset_trial_list(name, trials, pca_models)
        subset_embs[name] = embs
        subset_meta[name] = metas
        print(f"  {name:45s}: {len(embs):4d} 条 trial")
    return subset_embs, subset_meta

# ═══════════════════════════════════════════════════════
# 核心：新 SIC-Real 计算
# ═══════════════════════════════════════════════════════
def sic_real(emb_list, anchors, d_bar_side, d_bar_wrist, lam=BEST_LAMBDA,
             no_side=False, no_wrist=False, no_sat=False,
             kernel_mode=BEST_KERNEL):
    """
    emb_list: list of {'side_first': ndarray, 'wrist_repr': ndarray}
    每条 trial 权重相等，无 τ 折扣。
    """
    sic_s, sic_w = 0.0, 0.0
    for a_val in anchors.values():
        sigma_s, sigma_w = 0.0, 0.0
        for feat in emb_list:
            if not no_side:
                sigma_s += compute_kernel(
                    a_val["side_first"], feat["side_first"],
                    d_bar_side, mode=kernel_mode)
            if not no_wrist:
                sigma_w += compute_kernel(
                    a_val["wrist_repr"], feat["wrist_repr"],
                    d_bar_wrist, mode=kernel_mode)
        if not no_side:
            sic_s += sigma_s if no_sat else sat(sigma_s)
        if not no_wrist:
            sic_w += sigma_w if no_sat else sat(sigma_w)
    return sic_s + lam * sic_w

def score_all_subsets(subset_embs, anchors, d_bar_side, d_bar_wrist,
                      lam=BEST_LAMBDA, no_side=False, no_wrist=False,
                      no_sat=False, kernel_mode=BEST_KERNEL):
    names, scores, srs = [], [], []
    for name, sr in SUCCESS_RATE.items():
        if name not in subset_embs:
            continue
        s = sic_real(subset_embs[name], anchors, d_bar_side, d_bar_wrist,
                     lam=lam, no_side=no_side, no_wrist=no_wrist,
                     no_sat=no_sat, kernel_mode=kernel_mode)
        names.append(name)
        scores.append(s)
        srs.append(sr)
    return names, np.array(scores), np.array(srs)

# ═══════════════════════════════════════════════════════
# 分析 1：相关性分析
# ═══════════════════════════════════════════════════════
def analysis_correlation(subset_embs, anchors, d_bar_side, d_bar_wrist, cn_font):
    print("\n[分析1] 相关性分析...")
    names, scores, srs = score_all_subsets(
        subset_embs, anchors, d_bar_side, d_bar_wrist)
    rho, p_val, tau = correlate(scores.tolist(), srs.tolist())
    print(f"  Spearman ρ = {rho:.4f},  Kendall τ = {tau:.4f},  p = {p_val:.4f}")

    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(7.5, 6))

    sc = ax.scatter(scores_norm, srs, c=srs, cmap="RdYlGn",
                    s=100, zorder=4, edgecolors="k", linewidths=0.6)

    # 拟合线
    z = np.polyfit(scores_norm, srs, 1)
    x_line = np.linspace(-0.02, 1.05, 100)
    ax.plot(x_line, np.poly1d(z)(x_line), "b--", linewidth=1.6, alpha=0.65,
            label="线性拟合")

    # 点标签
    for n, sx, sy in zip(names, scores_norm, srs):
        cn = SUBSET_CN.get(n, n)
        ax.annotate(cn, (sx, sy), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, fontproperties=cn_font,
                    color="#333333")

    # n=12 显著性参考线（水平方向无意义，此处在 ρ 信息框里说明）
    info = (f"Spearman ρ = {rho:.4f}\n"
            f"Kendall  τ = {tau:.4f}\n"
            f"p 值      = {p_val:.4f}\n"
            f"n = {len(names)}\n"
            f"─────────────\n"
            f"n=12 显著性参考\n"
            f"ρ>0.59 → p<0.05\n"
            f"ρ>0.73 → p<0.01")
    ax.text(0.02, 0.97, info, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.9))

    plt.colorbar(sc, ax=ax, label="成功率")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.15)
    ax.legend(fontsize=9, prop=cn_font)
    ax_set_cn_labels(ax, xlabel="SIC-Real 得分（归一化）",
                     ylabel="实测成功率", cn_font=cn_font)

    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "1_correlation.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  已保存: {out}")

    df = pd.DataFrame({
        "subset": names,
        "subset_cn": [SUBSET_CN.get(n, n) for n in names],
        "sic_score": scores,
        "sic_score_norm": scores_norm,
        "success_rate": srs,
        "spearman_rho": rho,
        "kendall_tau": tau,
        "p_value": p_val,
    })
    df.to_csv(os.path.join(SAVE_DIR, "1_correlation.csv"),
              index=False, encoding="utf-8-sig")
    return rho, tau, p_val, names, scores, srs

# ═══════════════════════════════════════════════════════
# 分析 2：留一法 LOO
# ═══════════════════════════════════════════════════════
def analysis_loo(subset_embs, anchors, d_bar_side, d_bar_wrist, cn_font):
    print("\n[分析2] 留一法相关性分析 (LOO)...")
    names_all, scores_all, srs_all = score_all_subsets(
        subset_embs, anchors, d_bar_side, d_bar_wrist)
    rho_full, _, tau_full = correlate(scores_all.tolist(), srs_all.tolist())
    print(f"  完整集 ρ = {rho_full:.4f}")

    loo_records = []
    for i, left_out in enumerate(names_all):
        idx = [j for j in range(len(names_all)) if j != i]
        rho_i, p_i, tau_i = correlate(
            scores_all[idx].tolist(), srs_all[idx].tolist())
        delta = rho_i - rho_full
        loo_records.append({
            "left_out": left_out,
            "left_out_cn": SUBSET_CN.get(left_out, left_out),
            "rho_loo": rho_i,
            "tau_loo": tau_i,
            "p_loo": p_i,
            "delta_rho": delta,
        })
        print(f"  去掉 {left_out:45s} → ρ={rho_i:.4f} (Δ={delta:+.4f})")

    df_loo = pd.DataFrame(loo_records).sort_values("delta_rho")
    df_loo.to_csv(os.path.join(SAVE_DIR, "2_loo.csv"),
                  index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    cn_names  = df_loo["left_out_cn"].tolist()
    delta_rho = df_loo["delta_rho"].tolist()
    colors    = ["#d73027" if d > 0 else "#4575b4" for d in delta_rho]

    bars = ax.barh(range(len(cn_names)), delta_rho,
                   color=colors, edgecolor="k", linewidth=0.5, height=0.6)
    ax.set_yticks(range(len(cn_names)))
    ax.set_yticklabels(cn_names, fontproperties=cn_font, fontsize=9)

    for bar, val in zip(bars, delta_rho):
        ax.text(val + (0.0008 if val >= 0 else -0.0008),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=8.5)

    ax.axvline(0, color="black", linewidth=1.2)
    margin = max(abs(min(delta_rho)), abs(max(delta_rho))) * 0.25
    ax.set_xlim(min(delta_rho) - margin, max(delta_rho) + margin)

    patch_r = mpatches.Patch(color="#d73027", label="排除后ρ升高（该点拉低相关）")
    patch_b = mpatches.Patch(color="#4575b4", label="排除后ρ降低（该点贡献相关）")
    ax.legend(handles=[patch_r, patch_b], fontsize=8.5,
              prop=cn_font, loc="lower right")

    ax.text(0.98, 0.02, f"完整集 ρ = {rho_full:.4f}",
            transform=ax.transAxes, fontsize=9, ha="right",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))
    ax.grid(axis="x", alpha=0.3)
    ax_set_cn_labels(ax, xlabel="Δρ（留一后 - 完整集）",
                     ylabel="被排除的子集", cn_font=cn_font)

    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "2_loo.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  已保存: {out}")
    return df_loo

# ═══════════════════════════════════════════════════════
# 分析 3：消融实验
# ═══════════════════════════════════════════════════════
def analysis_ablation(subset_embs, anchors, d_bar_side, d_bar_wrist, cn_font):
    print("\n[分析3] 消融实验...")

    configs = [
        dict(label="完整SIC-Real",  no_side=False, no_wrist=False, no_sat=False),
        dict(label="去掉侧视角",     no_side=True,  no_wrist=False, no_sat=False),
        dict(label="去掉腕部",       no_side=False, no_wrist=True,  no_sat=False),
        dict(label="去掉饱和函数",   no_side=False, no_wrist=False, no_sat=True),
        dict(label="仅侧视角",       no_side=False, no_wrist=True,  no_sat=False),
        dict(label="仅腕部",         no_side=True,  no_wrist=False, no_sat=False),
    ]

    records = []
    scatter_data = {}
    for cfg in configs:
        label = cfg.pop("label")
        names, scores, srs = score_all_subsets(
            subset_embs, anchors, d_bar_side, d_bar_wrist, **cfg)
        rho, p_val, tau = correlate(scores.tolist(), srs.tolist())
        records.append({"配置": label, "Spearman ρ": round(rho, 4),
                        "Kendall τ": round(tau, 4), "p 值": round(p_val, 4)})
        scatter_data[label] = (scores, srs, names)
        print(f"  {label:14s} → ρ={rho:.4f}, τ={tau:.4f}, p={p_val:.4f}")
        cfg["label"] = label  # 还原

    df_abl = pd.DataFrame(records)
    df_abl.to_csv(os.path.join(SAVE_DIR, "3_ablation.csv"),
                  index=False, encoding="utf-8-sig")

    # 图1：柱状图
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [r["配置"] for r in records]
    rhos   = [r["Spearman ρ"] for r in records]
    bar_colors = ["#2166ac"] + ["#d6604d"] * (len(labels) - 1)
    bars = ax.bar(range(len(labels)), rhos, color=bar_colors,
                  edgecolor="k", linewidth=0.6, width=0.55)
    ax.axhline(rhos[0], color="#2166ac", linestyle="--",
               linewidth=1.2, alpha=0.55, label="完整模型基线")

    # n=12 显著性参考线
    for sig_label, sig_val in SIG_LEVELS.items():
        ax.axhline(sig_val, color="gray", linestyle=":",
                   linewidth=0.9, alpha=0.7)
        ax.text(len(labels) - 0.4, sig_val + 0.005, sig_label,
                fontsize=7.5, color="gray", va="bottom")

    for bar, val in zip(bars, rhos):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
                f"{val:.4f}", ha="center", fontsize=9)

    ax.set_ylim(0, 1.08)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontproperties=cn_font, fontsize=9.5,
                       rotation=15, ha="right")
    ax.legend(fontsize=9, prop=cn_font)
    ax.grid(axis="y", alpha=0.3)
    ax_set_cn_labels(ax, ylabel="Spearman ρ", cn_font=cn_font)

    plt.tight_layout()
    out1 = os.path.join(SAVE_DIR, "3_ablation_bar.png")
    plt.savefig(out1, dpi=150)
    plt.close()

    # 图2：各配置散点矩阵
    ncols = 3
    nrows = (len(configs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, nrows * 4))
    axes_flat = axes.flatten()

    for idx, cfg in enumerate(configs):
        label = cfg["label"]
        ax = axes_flat[idx]
        scores, srs, names = scatter_data[label]
        sn = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        rho, _, _ = correlate(scores.tolist(), srs.tolist())

        ax.scatter(sn, srs, c=srs, cmap="RdYlGn", s=60,
                   zorder=3, edgecolors="k", linewidths=0.4)
        if sn.std() > 1e-6:
            z = np.polyfit(sn, srs, 1)
            xl = np.linspace(sn.min(), sn.max(), 50)
            ax.plot(xl, np.poly1d(z)(xl), "b--", linewidth=1.1, alpha=0.6)

        ax.text(0.05, 0.93, f"ρ = {rho:.4f}", transform=ax.transAxes,
                fontsize=9.5,
                bbox=dict(facecolor="lightyellow", alpha=0.85, boxstyle="round"))
        ax.set_title(label, fontsize=10, fontproperties=cn_font, pad=5)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(-0.05, 1.1)

    for j in range(len(configs), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.supxlabel("SIC-Real 得分（归一化）", fontproperties=cn_font, fontsize=11)
    fig.supylabel("实测成功率", fontproperties=cn_font, fontsize=11)
    plt.tight_layout()
    out2 = os.path.join(SAVE_DIR, "3_ablation_scatter.png")
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"  已保存: {out1}, {out2}")
    return df_abl

# ═══════════════════════════════════════════════════════
# 分析 4：超参数分析（只有 λ）
# ═══════════════════════════════════════════════════════
def analysis_hyperparams(subset_embs, anchors, d_bar_side, d_bar_wrist, cn_font):
    print("\n[分析4] 超参数分析 (λ 扫描)...")

    lam_list = [0.0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0,
                3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
    rho_list, tau_list, p_list = [], [], []

    for lam in lam_list:
        names, scores, srs = score_all_subsets(
            subset_embs, anchors, d_bar_side, d_bar_wrist, lam=lam)
        rho, p_val, tau = correlate(scores.tolist(), srs.tolist())
        rho_list.append(rho)
        tau_list.append(tau)
        p_list.append(p_val)
        print(f"  λ={lam:5.1f} → ρ={rho:.4f}, τ={tau:.4f}, p={p_val:.4f}")

    best_idx = int(np.argmax(rho_list))
    best_lam = lam_list[best_idx]
    print(f"  最优 λ = {best_lam},  ρ = {rho_list[best_idx]:.4f}")

    # 核函数对比（λ=best_lam）
    kernel_results = {}
    for km in ["rbf", "cosine"]:
        names, scores, srs = score_all_subsets(
            subset_embs, anchors, d_bar_side, d_bar_wrist,
            lam=best_lam, kernel_mode=km)
        rho, p_val, tau = correlate(scores.tolist(), srs.tolist())
        kernel_results[km] = (rho, tau, p_val)
        print(f"  kernel={km:6s} → ρ={rho:.4f}, τ={tau:.4f}")

    # 图：λ 扫描曲线 + 显著性参考线
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(lam_list, rho_list, "o-", color="#d73027", linewidth=2,
            markersize=7, label="Spearman ρ", zorder=4)
    ax.plot(lam_list, tau_list, "s--", color="#4575b4", linewidth=1.8,
            markersize=6, label="Kendall τ", zorder=4)

    # 显著性参考线
    for sig_label, sig_val in SIG_LEVELS.items():
        ax.axhline(sig_val, color="gray", linestyle=":",
                   linewidth=1.0, alpha=0.7)
        ax.text(lam_list[-1] * 1.01, sig_val,
                sig_label, fontsize=8, color="gray", va="center")

    # 最优 λ 标记
    ax.axvline(best_lam, color="green", linestyle="--",
               linewidth=1.5, alpha=0.7, label=f"最优 λ={best_lam}")
    ax.scatter([best_lam], [rho_list[best_idx]], s=150, color="green",
               zorder=5, marker="*")

    ax.set_xlim(-0.5, lam_list[-1] + 1.5)
    ax.set_ylim(max(0, min(rho_list) - 0.1), 1.05)
    ax.legend(fontsize=9.5, prop=cn_font)
    ax.grid(True, alpha=0.3)
    ax_set_cn_labels(ax, xlabel="λ（wrist 权重）",
                     ylabel="相关系数", cn_font=cn_font)

    # 添加 kernel 对比注释
    km_text = ("核函数对比 (λ={:.1f})\n"
               "  rbf:    ρ={:.4f}\n"
               "  cosine: ρ={:.4f}").format(
        best_lam,
        kernel_results["rbf"][0],
        kernel_results["cosine"][0])
    ax.text(0.98, 0.97, km_text, transform=ax.transAxes,
            fontsize=8.5, va="top", ha="right", family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.85))

    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "4_lambda_sweep.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  已保存: {out}")

    df_hp = pd.DataFrame({
        "lambda": lam_list,
        "spearman_rho": rho_list,
        "kendall_tau": tau_list,
        "p_value": p_list,
    })
    df_hp.to_csv(os.path.join(SAVE_DIR, "4_hyperparam.csv"),
                 index=False, encoding="utf-8-sig")
    return best_lam, rho_list[best_idx]

# ═══════════════════════════════════════════════════════
# 分析 5：噪声估计（同 (p,r) 下多 trial 嵌入的方差）
# ═══════════════════════════════════════════════════════
def analysis_noise_estimate(trials, pca_models, cn_font):
    print("\n[分析5] 噪声估计（同 (p,r) 下 trial 间嵌入差异）...")

    side_stds, wrist_stds = [], []
    side_dists, wrist_dists = [], []

    for p in range(1, 10):
        for r in range(8):
            keys = [(p, r, t) for t in range(1, 5) if (p, r, t) in trials]
            if len(keys) < 2:
                continue
            side_embs  = []
            wrist_embs = []
            for key in keys:
                emb = extract_trial_embedding(trials[key], pca_models)
                side_embs.append(emb["side_first"])
                wrist_embs.append(emb["wrist_repr"])

            side_embs  = np.array(side_embs)
            wrist_embs = np.array(wrist_embs)

            # 各维度标准差的均值
            side_stds.append(side_embs.std(axis=0).mean())
            wrist_stds.append(wrist_embs.std(axis=0).mean())

            # 所有 trial pair 之间的欧氏距离
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    side_dists.append(
                        np.linalg.norm(side_embs[i] - side_embs[j]))
                    wrist_dists.append(
                        np.linalg.norm(wrist_embs[i] - wrist_embs[j]))

    side_stds  = np.array(side_stds)
    wrist_stds = np.array(wrist_stds)
    side_dists  = np.array(side_dists)
    wrist_dists = np.array(wrist_dists)

    print(f"  side  嵌入差异: mean_std={side_stds.mean():.3f}, "
          f"mean_dist={side_dists.mean():.3f}")
    print(f"  wrist 嵌入差异: mean_std={wrist_stds.mean():.3f}, "
          f"mean_dist={wrist_dists.mean():.3f}")

    # 加载 d_bar 作为参照
    anchor_key = f"anchors_{BEST_DBAR}_{BEST_CPLX}"
    with open(os.path.join(CACHE_DIR, f"{anchor_key}.pkl"), "rb") as f:
        _, d_bar_side, d_bar_wrist = pickle.load(f)
    d_bar_side_corrected = 183.5
    d_bar_side=d_bar_side_corrected


    ratio_side  = side_dists.mean()  / d_bar_side
    ratio_wrist = wrist_dists.mean() / d_bar_wrist
    print(f"  side  trial间距 / d_bar_side  = {ratio_side:.3f}  "
          f"({'小' if ratio_side < 0.1 else '中' if ratio_side < 0.3 else '大'})")
    print(f"  wrist trial间距 / d_bar_wrist = {ratio_wrist:.3f}  "
          f"({'小' if ratio_wrist < 0.1 else '中' if ratio_wrist < 0.3 else '大'})")

    # 图：分布直方图
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, dists, stds, modality, d_bar in zip(
        axes,
        [side_dists, wrist_dists],
        [side_stds, wrist_stds],
        ["side（侧视角）", "wrist（腕部）"],
        [d_bar_side, d_bar_wrist],
    ):
        ax.hist(dists, bins=25, color="#4393c3", alpha=0.7,
                edgecolor="k", linewidth=0.5, label="trial 间欧氏距离")
        ax.axvline(dists.mean(), color="r", linewidth=1.8,
                   linestyle="--", label=f"均值={dists.mean():.2f}")
        ax.axvline(d_bar, color="g", linewidth=1.8,
                   linestyle=":", label=f"d_bar={d_bar:.2f}")
        ax.legend(fontsize=8, prop=cn_font)
        ax.grid(True, alpha=0.3)
        ax.set_title(modality, fontsize=10, fontproperties=cn_font)
        ax_set_cn_labels(ax, xlabel="欧氏距离", ylabel="频次", cn_font=cn_font)

        info = (f"均值距离={dists.mean():.3f}\n"
                f"d_bar={d_bar:.3f}\n"
                f"比值={dists.mean()/d_bar:.3f}")
        ax.text(0.97, 0.97, info, transform=ax.transAxes,
                fontsize=8.5, va="top", ha="right", family="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "5_noise_estimate.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  已保存: {out}")

    noise_info = {
        "side_mean_dist":  float(side_dists.mean()),
        "side_std_dist":   float(side_dists.std()),
        "wrist_mean_dist": float(wrist_dists.mean()),
        "wrist_std_dist":  float(wrist_dists.std()),
        "d_bar_side":      float(d_bar_side),
        "d_bar_wrist":     float(d_bar_wrist),
        "ratio_side":      float(ratio_side),
        "ratio_wrist":     float(ratio_wrist),
        "recommended_noise_side":  float(side_dists.mean()),
        "recommended_noise_wrist": float(wrist_dists.mean()),
    }
    pd.Series(noise_info).to_csv(
        os.path.join(SAVE_DIR, "5_noise_estimate.csv"), header=["value"],
        encoding="utf-8-sig")
    return noise_info

# ═══════════════════════════════════════════════════════
# 分析 6：新旧 SIC 对比
# ═══════════════════════════════════════════════════════
def analysis_compare_old_new(subset_embs, trials, pca_models,
                             anchors, d_bar_side, d_bar_wrist,
                             cn_font):
    print("\n[分析6] 新旧 SIC 对比...")

    # ── 旧 SIC-B（带 τ）─────────────────────────────────
    def _tau(t_idx, alpha):
        return alpha * np.log((t_idx + 1) / t_idx)

    def sic_old(subset_feats_grouped, anchors, d_bar_side, d_bar_wrist,
                alpha=0.05, lam=0.5, kernel_mode="rbf"):
        sic_s, sic_w = 0.0, 0.0
        for a_val in anchors.values():
            sigma_s, sigma_w = 0.0, 0.0
            for feats in subset_feats_grouped.values():
                for t_idx, feat in enumerate(feats, start=1):
                    dt = _tau(t_idx, alpha)
                    sigma_s += dt * compute_kernel(
                        a_val["side_first"], feat["side_first"],
                        d_bar_side, mode=kernel_mode)
                    sigma_w += dt * compute_kernel(
                        a_val["wrist_repr"], feat["wrist_repr"],
                        d_bar_wrist, mode=kernel_mode)
            sic_s += sat(sigma_s)
            sic_w += sat(sigma_w)
        return sic_s + lam * sic_w

    def build_grouped_feats(name, trials, pca_models):
        sd = get_subset_dict(name)
        grouped = {}
        for (p, r), T in sd.items():
            feats = []
            for t in range(1, T + 1):
                key = (p, r, t)
                if key not in trials:
                    continue
                feats.append(extract_trial_embedding(trials[key], pca_models))
            if feats:
                grouped[(p, r)] = feats
        return grouped

    # 计算两版分数
    old_scores, new_scores, srs_list, name_list = [], [], [], []
    for name, sr in SUCCESS_RATE.items():
        grouped = build_grouped_feats(name, trials, pca_models)
        if not grouped:
            continue
        s_old = sic_old(grouped, anchors, d_bar_side, d_bar_wrist)
        s_new = sic_real(subset_embs[name], anchors, d_bar_side, d_bar_wrist,
                         lam=BEST_LAMBDA)
        old_scores.append(s_old)
        new_scores.append(s_new)
        srs_list.append(sr)
        name_list.append(name)

    old_scores = np.array(old_scores)
    new_scores = np.array(new_scores)
    srs_arr    = np.array(srs_list)

    rho_old, p_old, tau_old = correlate(old_scores.tolist(), srs_arr.tolist())
    rho_new, p_new, tau_new = correlate(new_scores.tolist(), srs_arr.tolist())
    print(f"  旧 SIC-B  (带τ): ρ={rho_old:.4f}, τ={tau_old:.4f}, p={p_old:.4f}")
    print(f"  新 SIC-Real(无τ): ρ={rho_new:.4f}, τ={tau_new:.4f}, p={p_new:.4f}")

    # 图：左侧两个散点，右侧柱状对比
    fig = plt.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
    ax_old = fig.add_subplot(gs[0])
    ax_new = fig.add_subplot(gs[1])
    ax_bar = fig.add_subplot(gs[2])

    for ax, scores, rho, label in [
        (ax_old, old_scores, rho_old, "旧 SIC-B（带τ）"),
        (ax_new, new_scores, rho_new, "新 SIC-Real（无τ）"),
    ]:
        sn = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        ax.scatter(sn, srs_arr, c=srs_arr, cmap="RdYlGn",
                   s=70, zorder=3, edgecolors="k", linewidths=0.5)
        if sn.std() > 1e-6:
            z = np.polyfit(sn, srs_arr, 1)
            xl = np.linspace(-0.02, 1.05, 60)
            ax.plot(xl, np.poly1d(z)(xl), "b--", linewidth=1.3, alpha=0.6)
        ax.text(0.05, 0.93, f"ρ = {rho:.4f}", transform=ax.transAxes,
                fontsize=10,
                bbox=dict(facecolor="lightyellow", alpha=0.85, boxstyle="round"))
        ax.set_title(label, fontsize=10, fontproperties=cn_font, pad=5)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(-0.05, 1.1)
        ax_set_cn_labels(ax, xlabel="归一化得分", ylabel="实测成功率",
                         cn_font=cn_font)

    # 柱状对比
    metrics = ["Spearman ρ", "Kendall τ"]
    old_vals = [rho_old, tau_old]
    new_vals = [rho_new, tau_new]
    x = np.arange(len(metrics))
    w = 0.32
    bars_o = ax_bar.bar(x - w/2, old_vals, w, label="旧 SIC-B",
                        color="#fc8d59", edgecolor="k", linewidth=0.6)
    bars_n = ax_bar.bar(x + w/2, new_vals, w, label="新 SIC-Real",
                        color="#91bfdb", edgecolor="k", linewidth=0.6)
    for bar in list(bars_o) + list(bars_n):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.4f}",
                    ha="center", fontsize=8.5)
    for sig_label, sig_val in SIG_LEVELS.items():
        ax_bar.axhline(sig_val, color="gray", linestyle=":",
                       linewidth=0.9, alpha=0.65)
        ax_bar.text(1.45, sig_val + 0.005, sig_label,
                    fontsize=7, color="gray")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metrics, fontproperties=cn_font, fontsize=10)
    ax_bar.set_ylim(0, 1.1)
    ax_bar.legend(fontsize=9, prop=cn_font)
    ax_bar.grid(axis="y", alpha=0.3)
    ax_set_cn_labels(ax_bar, ylabel="相关系数", cn_font=cn_font)

    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "6_old_vs_new.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  已保存: {out}")

    df_cmp = pd.DataFrame({
        "subset": name_list,
        "subset_cn": [SUBSET_CN.get(n, n) for n in name_list],
        "success_rate": srs_list,
        "score_old": old_scores,
        "score_new": new_scores,
    })
    df_cmp["rho_old"] = rho_old
    df_cmp["rho_new"] = rho_new
    df_cmp.to_csv(os.path.join(SAVE_DIR, "6_old_vs_new.csv"),
                  index=False, encoding="utf-8-sig")
    return rho_old, rho_new

# ═══════════════════════════════════════════════════════
# 汇总报告
# ═══════════════════════════════════════════════════════
def save_summary(rho, tau, p_val, df_loo, df_abl,
                 best_lam, best_rho_lam, noise_info,
                 rho_old, rho_new):
    lines = [
        "=" * 62,
        "  SIC-Real 分析结果汇总",
        "  每条 trial 视为独立个体，无 τ 折扣",
        "=" * 62,
        f"  配置: d_bar={BEST_DBAR}, cplx={BEST_CPLX}, kernel={BEST_KERNEL}",
        f"  默认 λ = {BEST_LAMBDA}",
        "",
        "【n=12 显著性参考】",
        "  ρ > 0.503 → p < 0.10",
        "  ρ > 0.587 → p < 0.05 ✓",
        "  ρ > 0.727 → p < 0.01 ✓✓",
        "",
        "【1. 相关性分析（默认λ）】",
        f"  Spearman ρ = {rho:.4f}",
        f"  Kendall  τ = {tau:.4f}",
        f"  p 值      = {p_val:.4f}",
        "",
        "【2. LOO 分析】",
        f"  最影响相关性（拉低）: "
        f"{df_loo.iloc[-1]['left_out_cn']} "
        f"(Δρ={df_loo.iloc[-1]['delta_rho']:+.4f})",
        f"  最贡献相关性（贡献）: "
        f"{df_loo.iloc[0]['left_out_cn']} "
        f"(Δρ={df_loo.iloc[0]['delta_rho']:+.4f})",
        "",
        "【3. 消融实验】",
    ]
    for _, row in df_abl.iterrows():
        lines.append(f"  {row['配置']:14s}: ρ={row['Spearman ρ']:.4f}")
    lines += [
        "",
        "【4. 超参数 λ 扫描】",
        f"  最优 λ = {best_lam},  ρ = {best_rho_lam:.4f}",
        "",
        "【5. 噪声估计】",
        f"  side  trial间距 = {noise_info['side_mean_dist']:.3f}  "
        f"(d_bar的 {noise_info['ratio_side']:.1%})",
        f"  wrist trial间距 = {noise_info['wrist_mean_dist']:.3f}  "
        f"(d_bar的 {noise_info['ratio_wrist']:.1%})",
        f"  建议噪声σ_side  = {noise_info['recommended_noise_side']:.3f}",
        f"  建议噪声σ_wrist = {noise_info['recommended_noise_wrist']:.3f}",
        "",
        "【6. 新旧 SIC 对比】",
        f"  旧 SIC-B  (带τ): ρ = {rho_old:.4f}",
        f"  新 SIC-Real(无τ): ρ = {rho_new:.4f}",
        f"  差值 Δρ = {rho_new - rho_old:+.4f}",
        "=" * 62,
    ]
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(SAVE_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt)

# ═══════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("=" * 62)
    print("  SIC-Real 分析  —  每条 trial 独立，无 τ 折扣")
    print("=" * 62)

    cn_font = setup_chinese_font()
    trials, pca_models, anchors, d_bar_side, d_bar_wrist = load_cache()
    subset_embs, subset_meta = build_all_subset_embs(trials, pca_models)

    rho, tau, p_val, _, _, _ = analysis_correlation(
        subset_embs, anchors, d_bar_side, d_bar_wrist, cn_font)

    df_loo = analysis_loo(
        subset_embs, anchors, d_bar_side, d_bar_wrist, cn_font)

    df_abl = analysis_ablation(
        subset_embs, anchors, d_bar_side, d_bar_wrist, cn_font)

    best_lam, best_rho_lam = analysis_hyperparams(
        subset_embs, anchors, d_bar_side, d_bar_wrist, cn_font)

    noise_info = analysis_noise_estimate(trials, pca_models, cn_font)

    rho_old, rho_new = analysis_compare_old_new(
        subset_embs, trials, pca_models,
        anchors, d_bar_side, d_bar_wrist, cn_font)

    save_summary(rho, tau, p_val, df_loo, df_abl,
                 best_lam, best_rho_lam, noise_info,
                 rho_old, rho_new)

    print(f"\n[完成] 总耗时 {time.time() - t0:.1f}s")
    print(f"  结果保存至: {SAVE_DIR}/")

if __name__ == "__main__":
    main()