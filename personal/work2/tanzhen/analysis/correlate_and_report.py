"""
相关性分析与报告生成

计算覆盖空隙与脆弱度的Spearman相关性，生成可视化图表和Markdown报告。

新增功能：
- 轴间对比：noise和layout两个轴各自的平均成功率跌幅
- 空间弱点分布图：112格点按x-y坐标铺开的成功率跌幅热力图
- 覆盖空隙分析降级为附录
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def load_probe_results(raw_dir: str, checkpoint_name: str) -> Dict[str, List[Dict]]:
    """
    加载探针评测原始结果

    参数:
        raw_dir: probe_raw目录路径
        checkpoint_name: checkpoint名称

    返回:
        {condition: [result_dict, ...]} 字典
    """
    results = {}
    raw_path = Path(raw_dir)

    for condition_file in raw_path.glob(f"{checkpoint_name}_*.json"):
        # 跳过coverage_gaps文件
        if "coverage_gaps" in condition_file.stem:
            continue
        condition = condition_file.stem.replace(f"{checkpoint_name}_", "")
        with open(condition_file) as f:
            data = json.load(f)
        results[condition] = data.get("results", [])

    return results


def compute_success_rates(results: List[Dict]) -> Dict[int, float]:
    """
    计算每个格点的成功率

    参数:
        results: 该条件下的所有格点结果

    返回:
        {grid_id: success_rate}
    """
    rates = {}
    for r in results:
        rates[r["grid_id"]] = r["success_rate"]
    return rates


def compute_fragility(
    sr_nominal: Dict[int, float],
    sr_perturbed: Dict[int, float],
) -> Dict[int, float]:
    """
    计算脆弱度: fragility(c) = SR_nominal(c) - SR_perturbed(c)

    参数:
        sr_nominal: nominal条件下的成功率
        sr_perturbed: 扰动条件下的成功率

    返回:
        {grid_id: fragility}
    """
    fragility = {}
    for grid_id in sr_nominal:
        if grid_id in sr_perturbed:
            fragility[grid_id] = sr_nominal[grid_id] - sr_perturbed[grid_id]
    return fragility


def normalize_fragility(fragility: Dict[int, float]) -> Dict[int, float]:
    """
    归一化脆弱度到[0, 1]范围

    参数:
        fragility: 原始脆弱度

    返回:
        归一化后的脆弱度
    """
    values = list(fragility.values())
    if not values:
        return fragility

    min_val = min(values)
    max_val = max(values)

    if max_val == min_val:
        return {k: 0.5 for k in fragility}

    return {k: (v - min_val) / (max_val - min_val) for k, v in fragility.items()}


def compute_weakness_scores(
    fragility: Dict[int, float],
    beta: float = 1.0,
) -> Dict[str, float]:
    """
    计算脆弱度权重: w(c) = 1 + β · fragility_norm(c)

    参数:
        fragility: 原始脆弱度
        beta: 缩放系数

    返回:
        {"(x, y)": weakness_score} 字典（供v4读取）
    """
    fragility_norm = normalize_fragility(fragility)
    scores = {}
    for grid_id, frag in fragility_norm.items():
        coord_str = f"({grid_id}, 0)"  # 简化为1D索引，实际可根据需要调整
        scores[coord_str] = 1.0 + beta * frag
    return scores


def compute_correlation(
    gaps: Dict[int, float],
    fragility: Dict[int, float],
) -> Tuple[float, float]:
    """
    计算Spearman相关性

    参数:
        gaps: 覆盖空隙
        fragility: 脆弱度

    返回:
        (rho, p_value)
    """
    # 对齐两个字典的key
    common_keys = set(gaps.keys()) & set(fragility.keys())
    if not common_keys:
        return 0.0, 1.0

    gap_list = [gaps[k] for k in common_keys]
    frag_list = [fragility[k] for k in common_keys]

    rho, p_value = spearmanr(gap_list, frag_list)
    return rho, p_value


def plot_gap_vs_fragility(
    gaps: Dict[int, float],
    fragility: Dict[int, float],
    checkpoint_name: str,
    rho: float,
    p_value: float,
    output_path: str,
):
    """
    绘制覆盖空隙 vs 脆弱度散点图

    参数:
        gaps: 覆盖空隙
        fragility: 脆弱度
        checkpoint_name: checkpoint名称
        rho: Spearman相关系数
        p_value: p值
        output_path: 输出文件路径
    """
    common_keys = set(gaps.keys()) & set(fragility.keys())
    if not common_keys:
        return

    gap_list = [gaps[k] for k in common_keys]
    frag_list = [fragility[k] for k in common_keys]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(gap_list, frag_list, alpha=0.6, edgecolors="k", linewidth=0.5)

    # 添加趋势线
    z = np.polyfit(gap_list, frag_list, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(gap_list), max(gap_list), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8)

    ax.set_xlabel("Coverage Gap (distance to nearest anchor)", fontsize=12)
    ax.set_ylabel("Fragility (SR_nominal - SR_perturbed)", fontsize=12)
    ax.set_title(
        f"{checkpoint_name}: Gap vs Fragility\n"
        f"Spearman ρ={rho:.3f}, p={p_value:.4f}",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"散点图已保存到 {output_path}")


def plot_success_rate_by_condition(
    success_rates: Dict[str, Dict[int, float]],
    checkpoint_name: str,
    output_path: str,
):
    """
    绘制按条件分组的成功率柱状图

    参数:
        success_rates: {condition: {grid_id: rate}}
        checkpoint_name: checkpoint名称
        output_path: 输出文件路径
    """
    conditions = list(success_rates.keys())
    n_conditions = len(conditions)

    # 计算每个条件的平均成功率
    avg_rates = []
    for cond in conditions:
        rates = list(success_rates[cond].values())
        avg_rates.append(np.mean(rates) if rates else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71", "#e74c3c", "#c0392b", "#3498db", "#2980b9"]
    bars = ax.bar(conditions, avg_rates, color=colors[:n_conditions], edgecolor="black", linewidth=0.5)

    # 添加数值标签
    for bar, rate in zip(bars, avg_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{rate:.2%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Average Success Rate", fontsize=12)
    ax.set_title(f"{checkpoint_name}: Success Rate by Condition", fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"柱状图已保存到 {output_path}")


def plot_spatial_weakness_map(
    grid_points: List[Dict],
    fragility: Dict[int, float],
    axis_name: str,
    level: str,
    checkpoint_name: str,
    output_path: str,
):
    """
    绘制空间弱点分布图（按格点x-y坐标铺开的成功率跌幅热力图）

    参数:
        grid_points: 格点信息列表，包含 {"grid_id", "obj_pos"}
        fragility: 脆弱度字典 {grid_id: fragility}
        axis_name: 轴名称（noise或layout）
        level: 强度等级（L1或L2）
        checkpoint_name: checkpoint名称
        output_path: 输出文件路径
    """
    # 提取x-y坐标和脆弱度
    x_coords = []
    y_coords = []
    frag_values = []

    for gp in grid_points:
        grid_id = gp["grid_id"]
        if grid_id in fragility:
            x_coords.append(gp["obj_pos"][0])
            y_coords.append(gp["obj_pos"][1])
            frag_values.append(fragility[grid_id])

    if not x_coords:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # 使用散点图，颜色表示脆弱度
    scatter = ax.scatter(
        x_coords, y_coords,
        c=frag_values,
        cmap="RdYlGn_r",  # 红-黄-绿反转（红色=高脆弱度）
        vmin=0, vmax=1,
        s=100,
        edgecolors="black",
        linewidth=0.5,
        alpha=0.8,
    )

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, label="Fragility (SR_nominal - SR_perturbed)")
    cbar.ax.set_ylabel("Fragility", fontsize=12)

    ax.set_xlabel("Object X Position", fontsize=12)
    ax.set_ylabel("Object Y Position", fontsize=12)
    ax.set_title(
        f"{checkpoint_name}: {axis_name}-{level} Spatial Weakness Map\n"
        f"Red = High Fragility, Green = Low Fragility",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"空间弱点分布图已保存到 {output_path}")


def compute_axis_comparison(
    success_rates: Dict[str, Dict[int, float]],
) -> Dict[str, float]:
    """
    计算轴间对比：每个轴的平均成功率跌幅

    参数:
        success_rates: {condition: {grid_id: rate}}

    返回:
        {axis_name: avg_fragility}
    """
    sr_nominal = success_rates.get("nominal", {})
    if not sr_nominal:
        return {}

    # 计算noise轴的跌幅
    noise_l1 = success_rates.get("noise-L1", {})
    noise_l2 = success_rates.get("noise-L2", {})
    
    fragility_noise_l1 = compute_fragility(sr_nominal, noise_l1)
    fragility_noise_l2 = compute_fragility(sr_nominal, noise_l2)

    # 计算layout轴的跌幅
    layout_l1 = success_rates.get("layout-L1", {})
    layout_l2 = success_rates.get("layout-L2", {})
    
    fragility_layout_l1 = compute_fragility(sr_nominal, layout_l1)
    fragility_layout_l2 = compute_fragility(sr_nominal, layout_l2)

    return {
        "noise-L1": np.mean(list(fragility_noise_l1.values())) if fragility_noise_l1 else 0,
        "noise-L2": np.mean(list(fragility_noise_l2.values())) if fragility_noise_l2 else 0,
        "layout-L1": np.mean(list(fragility_layout_l1.values())) if fragility_layout_l1 else 0,
        "layout-L2": np.mean(list(fragility_layout_l2.values())) if fragility_layout_l2 else 0,
    }


def generate_report(
    config: dict,
    checkpoint_name: str,
    grid_points: List[Dict],
    success_rates: Dict[str, Dict[int, float]],
    fragilities: Dict[str, Dict[int, float]],
    gaps: Dict[int, float],
    correlations: Dict[str, Tuple[float, float]],
    weakness_scores: Dict[str, float],
    figures_dir: str,
    report_file: str,
    weakness_file: str,
):
    """
    生成实验报告

    参数:
        config: 实验配置
        checkpoint_name: checkpoint名称
        grid_points: 格点信息列表
        success_rates: {condition: {grid_id: rate}}
        fragilities: {axis_level: {grid_id: fragility}}
        gaps: {grid_id: gap}
        correlations: {axis_level: (rho, p_value)}
        weakness_scores: {grid_id: score}
        figures_dir: 图表目录
        report_file: 报告输出路径
        weakness_file: 脆弱度权重文件路径
    """
    report_lines = []

    def add(line: str):
        report_lines.append(line)

    add("# 探针实验报告 (Tanzhen Probe Experiment)")
    add("")
    add(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add("")

    # 实验配置
    add("## 1. 实验配置")
    add("")
    add(f"- **测试的Checkpoint**: {checkpoint_name}")
    add(f"- **为什么这轮只用一个checkpoint**: 先跑通pipeline，确认代码无误后再补第二个checkpoint做横向对比")
    add(f"- **扰动轴**:")
    if config["perturbations"]["noise"]["enabled"]:
        noise_levels = list(config["perturbations"]["noise"]["levels"].keys())
        add(f"  - Sensor Noise: {', '.join(noise_levels)}")
        add(f"    - L1: 高斯模糊 5×5, σ=1.5")
        add(f"    - L2: 高斯模糊 9×9, σ=3.0")
    if config["perturbations"].get("layout", {}).get("enabled", False):
        layout_levels = list(config["perturbations"]["layout"]["levels"].keys())
        add(f"  - Layout (Goal Position): {', '.join(layout_levels)}")
        add(f"    - L1: 目标位置偏移 ±0.02m")
        add(f"    - L2: 目标位置偏移 ±0.04m")
    if not config["perturbations"]["camera"]["enabled"]:
        add(f"  - Camera Viewpoint: **已废弃**（对固定摄像头的单任务部署没有诊断价值）")
    add(f"- **每格点Rollout次数**: {config['rollout']['n_repeats_final']} (正式版) / {config['rollout']['n_repeats_smoke']} (冒烟测试)")
    add(f"- **β值 (权重缩放系数)**: {config['analysis']['beta']}")
    add("")

    # 主要结果：轴间跌幅对比
    add("## 2. 主要结果")
    add("")
    add("### 2.1 轴间跌幅对比")
    add("")

    axis_comparison = compute_axis_comparison(success_rates)
    if axis_comparison:
        add("| 轴/强度 | 平均成功率跌幅 |")
        add("|---------|---------------|")
        for axis_level, avg_frag in axis_comparison.items():
            add(f"| {axis_level} | {avg_frag:.4f} |")
        add("")

        # 找出跌幅最大的轴
        max_axis = max(axis_comparison, key=axis_comparison.get)
        add(f"**跌幅最大的轴**: {max_axis} ({axis_comparison[max_axis]:.4f})")
        add("")
        add("> **免责声明**: 两个轴的L2强度不是严格对等强度（噪声模糊 vs 目标位置偏移），")
        add("> 这个排序只是方向性参考，不是精确定量结论。")
    else:
        add("（无数据）")
    add("")

    # 成功率汇总
    add("### 2.2 各条件成功率汇总")
    add("")
    if success_rates:
        add("| 条件 | 平均成功率 |")
        add("|------|-----------|")
        for cond in ["nominal", "noise-L1", "noise-L2", "layout-L1", "layout-L2"]:
            if cond in success_rates:
                rates = success_rates[cond]
                avg_rate = np.mean(list(rates.values())) if rates else 0
                add(f"| {cond} | {avg_rate:.2%} |")
    else:
        add("（无数据）")
    add("")

    # 柱状图
    bar_path = Path(figures_dir) / f"success_rate_by_condition_{checkpoint_name}.png"
    if bar_path.exists():
        add(f"![成功率柱状图]({bar_path})")
        add("")

    # 空间弱点分布图
    add("### 2.3 空间弱点分布图")
    add("")

    for axis_name in ["noise", "layout"]:
        for level in ["L1", "L2"]:
            condition = f"{axis_name}-{level}"
            if condition in fragilities:
                add(f"#### {condition}")
                add("")
                spatial_map_path = Path(figures_dir) / f"spatial_weakness_{checkpoint_name}_{condition}.png"
                if spatial_map_path.exists():
                    add(f"![{condition}空间弱点分布]({spatial_map_path})")
                    add("")
                else:
                    add(f"（图未生成）")
                    add("")

    # 弱区重合分析
    add("### 2.4 弱区重合分析")
    add("")
    noise_l2_frag = fragilities.get("noise-L2", {})
    layout_l2_frag = fragilities.get("layout-L2", {})

    if noise_l2_frag and layout_l2_frag:
        # 找出每个轴的弱点格点（脆弱度 > 0.5）
        noise_weak = {gid for gid, frag in noise_l2_frag.items() if frag > 0.5}
        layout_weak = {gid for gid, frag in layout_l2_frag.items() if frag > 0.5}

        overlap = noise_weak & layout_weak
        overlap_ratio = len(overlap) / max(len(noise_weak | layout_weak), 1)

        if overlap_ratio > 0.5:
            add(f"**弱区重合度**: {overlap_ratio:.1%} ({len(overlap)}/{len(noise_weak | layout_weak)} 个格点)")
            add("")
            add("两个轴的弱区高度重合，**可能和该区域训练数据覆盖不足有关**，见附录gap分析。")
        elif overlap_ratio > 0.2:
            add(f"**弱区重合度**: {overlap_ratio:.1%} ({len(overlap)}/{len(noise_weak | layout_weak)} 个格点)")
            add("")
            add("两个轴的弱区部分重合，说明既有覆盖不足的影响，也有独立的脆弱模式。")
        else:
            add(f"**弱区重合度**: {overlap_ratio:.1%} ({len(overlap)}/{len(noise_weak | layout_weak)} 个格点)")
            add("")
            add("**两个轴的脆弱模式相互独立，和覆盖度无关**。需要分别针对noise和layout优化。")
    else:
        add("（无数据）")
    add("")

    # 附录：覆盖空隙相关性分析
    add("## 3. 附录：覆盖空隙 vs 脆弱度相关性分析")
    add("")
    add("> **注意**: 当前只有一个checkpoint的初步观察，尚不能下验证通过/不通过的结论。")
    add("> 需第二个checkpoint补上后，按原计划要求两个checkpoint都显著才算稳健。")
    add("")

    for axis_level, (rho, p_value) in correlations.items():
        add(f"### {axis_level}")
        add("")
        add(f"- **Spearman ρ**: {rho:.4f}")
        add(f"- **p值**: {p_value:.6f}")
        significant = rho > config["analysis"]["rho_threshold"] and p_value < config["analysis"]["significance_threshold"]
        add(f"- **显著性**: {'是' if significant else '否'} (ρ>{config['analysis']['rho_threshold']} 且 p<{config['analysis']['significance_threshold']})")
        add("")

        # 散点图路径
        scatter_path = Path(figures_dir) / f"gap_vs_fragility_{checkpoint_name}_{axis_level}.png"
        if scatter_path.exists():
            add(f"![散点图]({scatter_path})")
            add("")

    # weakness_scores.json信息
    add("## 4. weakness_scores.json 信息")
    add("")
    add(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"- **文件路径**: `{weakness_file}`")

    if weakness_scores:
        values = list(weakness_scores.values())
        add(f"- **取值范围**: [{min(values):.4f}, {max(values):.4f}]")
        add(f"- **平均值**: {np.mean(values):.4f}")
        add(f"- **格点数量**: {len(values)}")
    else:
        add("（无数据）")

    add("")

    # 待核实项确认结果
    add("## 5. 待核实项确认结果")
    add("")
    add("### 5.1 Checkpoint可用性")
    add("")
    add("- **random_42**: `personal/work2/duibi/random_42/random_112_seed42/checkpoints/000200/pretrained_model/` - 已确认存在")
    add("- **uniform_42**: `personal/work2/duibi/uniform_42/uniform_112_seed42/checkpoints/last/pretrained_model/` - 已确认存在")
    add("")
    add("### 5.2 Layout扰动方案")
    add("")
    add("- **选定方案**: 方案A - 扰动goal位置（保持物体格点位置不变）")
    add("- **实现方式**: 修改 `env.goal` 和 `env._get_state_rand_vec`")
    add("- **L1**: 目标位置偏移 ±0.02m")
    add("- **L2**: 目标位置偏移 ±0.04m")
    add("")
    add("### 5.3 历史结果处理")
    add("")
    add("- 之前 `results/probe_raw/` 里的历史结果是在占位符/随机动作状态下生成的，已标记无效")
    add("- 重命名加 `_INVALID_RANDOM_ACTION` 后缀或移到 `_deprecated/`")
    add("")

    add("---")
    add("*本报告由 `correlate_and_report.py` 自动生成*")

    # 写入文件
    report_path = Path(report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"实验报告已保存到 {report_path}")


def main():
    parser = argparse.ArgumentParser(description="相关性分析与报告生成")
    parser.add_argument("--config", type=str, default="personal/work2/tanzhen/configs/probe_config.yaml")
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    raw_dir = args.raw_dir or config["output"]["raw_dir"]
    figures_dir = config["output"]["figures_dir"]
    report_file = config["output"]["report_file"]
    weakness_file = config["output"]["weakness_file"]

    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    # 只使用active_checkpoint
    checkpoint_name = config.get("active_checkpoint", "random_42")
    ckpt_config = config["checkpoints"][checkpoint_name]

    print(f"\n分析Checkpoint: {checkpoint_name}")

    # 加载探针结果
    results = load_probe_results(raw_dir, checkpoint_name)
    if not results:
        print(f"  警告: 未找到 {checkpoint_name} 的探针结果")
        return

    # 计算各条件成功率
    success_rates = {}
    for cond, res in results.items():
        success_rates[cond] = compute_success_rates(res)

    # 加载格点信息
    subset_file = ckpt_config["subset_file"]
    dataset_metadata = config["dataset_metadata"]
    with open(subset_file) as f:
        subset_data = json.load(f)
    with open(dataset_metadata) as f:
        metadata = json.load(f)
    episodes = metadata["episodes"]

    if isinstance(subset_data, list):
        episode_indices = subset_data
    elif isinstance(subset_data, dict):
        episode_indices = subset_data.get("selected_episode_indices", 
                           subset_data.get("episodes", 
                           subset_data.get("included_episodes", [])))
    else:
        episode_indices = []

    grid_points = []
    for idx in episode_indices:
        episode = episodes[idx]
        grid_points.append({
            "grid_id": idx,
            "obj_pos": episode["obj_init_pos"],
            "goal_pos": episode.get("goal_pose", [0.1, 0.8, 0.2]),
        })

    # 计算脆弱度（每个轴每个强度）
    sr_nominal = success_rates.get("nominal", {})
    fragilities = {}

    for cond in ["noise-L1", "noise-L2", "layout-L1", "layout-L2"]:
        if cond in success_rates:
            fragilities[cond] = compute_fragility(sr_nominal, success_rates[cond])

    # 综合脆弱度（noise和layout的均值）
    fragility_combined = {}
    for grid_id in sr_nominal:
        frags = []
        for cond in ["noise-L1", "noise-L2", "layout-L1", "layout-L2"]:
            if cond in fragilities and grid_id in fragilities[cond]:
                frags.append(fragilities[cond][grid_id])
        if frags:
            fragility_combined[grid_id] = np.mean(frags)

    # 加载覆盖空隙
    gaps_file = Path(raw_dir) / f"{checkpoint_name}_coverage_gaps.json"
    if gaps_file.exists():
        with open(gaps_file) as f:
            gaps = {int(k): v for k, v in json.load(f).items()}
    else:
        print(f"  警告: 未找到 {checkpoint_name} 的覆盖空隙数据")
        gaps = {}

    # 计算相关性（每个轴分别计算）
    correlations = {}
    for cond in ["noise-L1", "noise-L2", "layout-L1", "layout-L2"]:
        if cond in fragilities and gaps:
            rho, p_value = compute_correlation(gaps, fragilities[cond])
            correlations[cond] = (rho, p_value)
            print(f"  {cond}: Spearman ρ={rho:.4f}, p={p_value:.6f}")

            # 绘制散点图
            scatter_path = Path(figures_dir) / f"gap_vs_fragility_{checkpoint_name}_{cond}.png"
            plot_gap_vs_fragility(gaps, fragilities[cond], checkpoint_name, rho, p_value, str(scatter_path))

    # 绘制成功率柱状图
    bar_path = Path(figures_dir) / f"success_rate_by_condition_{checkpoint_name}.png"
    plot_success_rate_by_condition(success_rates, checkpoint_name, str(bar_path))

    # 绘制空间弱点分布图
    for cond in ["noise-L1", "noise-L2", "layout-L1", "layout-L2"]:
        if cond in fragilities:
            axis_name = cond.split("-")[0]
            level = cond.split("-")[1]
            spatial_map_path = Path(figures_dir) / f"spatial_weakness_{checkpoint_name}_{cond}.png"
            plot_spatial_weakness_map(
                grid_points, fragilities[cond], axis_name, level, checkpoint_name, str(spatial_map_path)
            )

    # 计算脆弱度权重
    weakness_scores = compute_weakness_scores(fragility_combined, beta=config["analysis"]["beta"])

    # 保存weakness_scores.json
    weakness_path = Path(weakness_file)
    weakness_path.parent.mkdir(parents=True, exist_ok=True)
    with open(weakness_path, "w") as f:
        json.dump(weakness_scores, f, indent=2)
    print(f"\n脆弱度权重已保存到 {weakness_path}")

    # 生成报告
    generate_report(
        config=config,
        checkpoint_name=checkpoint_name,
        grid_points=grid_points,
        success_rates=success_rates,
        fragilities=fragilities,
        gaps=gaps,
        correlations=correlations,
        weakness_scores=weakness_scores,
        figures_dir=figures_dir,
        report_file=report_file,
        weakness_file=weakness_file,
    )


if __name__ == "__main__":
    main()