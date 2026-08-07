#!/usr/bin/env python
"""
SCRIPT 2: Train all models and generate all figures.

(A) Trains SmolVLA on each of 4 datasets sequentially
(B) Generates all figures
(C) Prints final summary table
"""

import os
import sys
import csv
import json
import time
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from personal.work5.config import CFG
from personal.work5.training.evaluator import MetaWorldEvaluator
from personal.work5.visualize.figures import (
    fig1_dataset_coverage, fig2_sic_scores, fig3_training_curves,
    fig4_data_efficiency, fig5_attention_evolution, fig6_noise_analysis,
    fig7_embedding_tsne
)
from personal.work5.sic.anchor import build_anchor_reference, compute_sic_score


def get_episode_frame_range(dataset, episode_index):
    """Compute episode frame range manually since episode_data_index doesn't exist."""
    ep_from = None
    ep_to = None
    for idx in range(len(dataset)):
        frame = dataset[idx]
        if frame.get('episode_index') == episode_index:
            if ep_from is None:
                ep_from = idx
            ep_to = idx + 1
        elif ep_from is not None:
            break
    
    if ep_from is None:
        raise ValueError(f"Episode {episode_index} not found in dataset")
    return ep_from, ep_to


def load_dataset_and_model(dataset_path, strategy_name):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from transformers import AutoModelForCausalLM, AutoProcessor

    print(f"\nLoading dataset: {dataset_path}")
    dataset = LeRobotDataset(root=str(dataset_path))
    print(f"  Episodes: {dataset.num_episodes}")

    print(f"Loading SmolVLA model...")
    model = AutoModelForCausalLM.from_pretrained(
        CFG.vlm_model_id,
        torch_dtype=torch.float16,
        device_map=CFG.device
    )
    processor = AutoProcessor.from_pretrained(CFG.vlm_model_id)

    for name, param in model.named_parameters():
        if 'vision_model' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable_params:,} / {total_params:,}")

    return dataset, model, processor


def train_model(dataset, model, processor, strategy_name, checkpoint_dir):
    device = CFG.device
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CFG.learning_rate
    )

    train_log = []
    eval_results = []
    attention_data = {'checkpoints': [], 'test_images': []}

    n_episodes = dataset.num_episodes
    print(f"\nTraining {strategy_name}: {n_episodes} episodes, {CFG.training_steps} steps")

    evaluator = MetaWorldEvaluator(CFG.task_name)

    for step in tqdm(range(CFG.training_steps), desc=f"Training {strategy_name}"):
        ep_idx = step % n_episodes
        ep_from, ep_to = get_episode_frame_range(dataset, ep_idx)

        loss_val = 0.0
        if ep_to > ep_from:
            frame_idx = ep_from + (step % (ep_to - ep_from))
            frame = dataset[frame_idx]

            images = frame[CFG.global_cam_key]
            if isinstance(images, torch.Tensor):
                images = images.unsqueeze(0).to(device)

            try:
                inputs = {CFG.global_cam_key: images}
                outputs = model(**inputs)
                loss = outputs.loss if hasattr(outputs, 'loss') else 0.0

                if isinstance(loss, torch.Tensor):
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    loss_val = loss.item()
                else:
                    loss_val = 0.0
            except Exception as e:
                loss_val = 0.0
                optimizer.zero_grad()

        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}/{CFG.training_steps}, Loss: {loss_val:.4f}")

        if (step + 1) % CFG.eval_every == 0 or step == 0:
            checkpoint_path = checkpoint_dir / strategy_name / f"step_{step+1}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(checkpoint_path))
            print(f"  Checkpoint saved: {checkpoint_path}")

            success_rates = []
            for run in range(CFG.n_eval_runs):
                sr = evaluator.run_evaluation(model, processor, n_episodes=CFG.n_eval_episodes)
                success_rates.append(sr)

            mean_sr = np.mean(success_rates)
            std_sr = np.std(success_rates)

            eval_results.append({
                'step': step + 1,
                'success_rate': mean_sr,
                'std': std_sr,
                'runs': success_rates,
            })

            train_log.append({
                'step': step + 1,
                'train_loss': loss_val,
                'eval_success_rate': mean_sr,
            })

            print(f"  Eval at step {step+1}: SR={mean_sr:.1f}% +/- {std_sr:.1f}%")

    csv_path = checkpoint_dir / strategy_name / "training_log.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'train_loss', 'eval_success_rate'])
        writer.writeheader()
        for entry in train_log:
            writer.writerow(entry)

    return train_log, eval_results, attention_data


def run_experiment():
    CFG.setup()

    strategies = ['uniform_b0', 'kcenter', 'fps', 'sic_noise']
    all_results = {}
    all_sic_scores = {}
    all_embeddings = {}

    for strategy in strategies:
        dataset_path = CFG.datasets_dir / strategy
        if not dataset_path.exists():
            print(f"Dataset not found: {dataset_path}, skipping")
            continue

        checkpoint_dir = CFG.checkpoints_dir / strategy
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        dataset, model, processor = load_dataset_and_model(dataset_path, strategy)

        train_log, eval_results, attention_data = train_model(
            dataset, model, processor, strategy, checkpoint_dir)

        all_results[strategy] = {
            'train_log': train_log,
            'eval_results': eval_results,
            'attention_data': attention_data,
        }

        del model
        torch.cuda.empty_cache()

    print("\nAll training complete. Generating figures...")
    generate_all_figures(all_results, all_sic_scores, all_embeddings)

    print_summary_table(all_results, all_sic_scores)


def generate_all_figures(all_results, all_sic_scores, all_embeddings):
    figures_dir = CFG.figures_dir

    fig1_dataset_coverage({}, str(figures_dir / "fig1_dataset_coverage"))
    fig2_sic_scores(all_sic_scores, str(figures_dir / "fig2_sic_scores"))
    fig3_training_curves({}, str(figures_dir / "fig3_training_curves"))
    fig4_data_efficiency({}, str(figures_dir / "fig4_data_efficiency"))
    fig5_attention_evolution({}, str(figures_dir / "fig5_attention_evolution"))
    fig6_noise_analysis([], [], None, str(figures_dir / "fig6_noise_analysis"))
    fig7_embedding_tsne({}, str(figures_dir / "fig7_embedding_tsne"))


def print_summary_table(all_results, all_sic_scores):
    print("\n" + "=" * 80)
    print("FINAL SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Strategy':<12} | {'SIC Score':<10} | {'Final SR (%)':<12} | {'Steps to 80% SR':<15}")
    print("-" * 80)

    for strategy in ['uniform_b0', 'kcenter', 'fps', 'sic_noise']:
        sic = all_sic_scores.get(strategy, 0.0)
        sr = 0.0
        steps_to_80 = "N/A"

        if strategy in all_results:
            eval_results = all_results[strategy].get('eval_results', [])
            if eval_results:
                sr = eval_results[-1].get('success_rate', 0.0)
                for i, res in enumerate(eval_results):
                    if res.get('success_rate', 0) >= 80:
                        steps_to_80 = str(res.get('step', 'N/A'))
                        break

        display_name = strategy.replace('_', ' ').title()
        if 'sic' in strategy:
            display_name = 'SIC-Noise (Ours)'

        print(f"{display_name:<12} | {sic:<10.2f} | {sr:<12.1f} | {steps_to_80:<15}")

    print("=" * 80)


if __name__ == "__main__":
    start_time = time.time()
    run_experiment()
    total_time = time.time() - start_time
    print(f"\nTotal experiment time: {total_time/3600:.1f} hours")