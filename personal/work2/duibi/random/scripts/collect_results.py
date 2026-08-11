#!/usr/bin/env python
"""
Collect and summarize results from random baseline experiments.
"""
import json
import os
from pathlib import Path

import numpy as np


def parse_eval_metrics(log_file: Path):
    """Parse evaluation success rate from training log."""
    if not log_file.exists():
        return None
    
    with open(log_file, "r") as f:
        content = f.read()
    
    # Look for eval success rate patterns
    success_rates = []
    for line in content.split("\n"):
        if "eval" in line.lower() and "success" in line.lower():
            # Try to extract success rate
            try:
                if "success_rate" in line:
                    parts = line.split("success_rate")
                    if len(parts) > 1:
                        rate_str = parts[1].strip().split()[0].replace(",", "")
                        rate = float(rate_str)
                        success_rates.append(rate)
            except (ValueError, IndexError):
                pass
    
    if success_rates:
        return {
            "final_success_rate": success_rates[-1],
            "all_success_rates": success_rates,
            "max_success_rate": max(success_rates),
        }
    return None


def summarize_results():
    """Summarize all random baseline experiment results."""
    log_dir = Path(__file__).parent.parent / "logs"
    
    print("=" * 70)
    print("Random Baseline Results Summary")
    print("=" * 70)
    print()
    
    episode_sizes = [100, 200, 300]
    seeds = [42, 142, 242]
    
    results = {}
    
    for num_eps in episode_sizes:
        results[num_eps] = {}
        for seed in seeds:
            exp_name = f"random_{num_eps}_seed{seed}"
            log_file = log_dir / f"{exp_name}.log"
            
            metrics = parse_eval_metrics(log_file)
            results[num_eps][seed] = metrics
            
            status = "✓" if metrics else "✗"
            if metrics:
                print(f"{status} {exp_name}:")
                print(f"    Final success rate: {metrics['final_success_rate']:.4f}")
                print(f"    Max success rate:   {metrics['max_success_rate']:.4f}")
            else:
                print(f"{status} {exp_name}: No results yet (log not found or incomplete)")
    
    print()
    print("=" * 70)
    print("Average Success Rates by Episode Size")
    print("=" * 70)
    
    for num_eps in episode_sizes:
        rates = []
        for seed in seeds:
            if results[num_eps][seed] is not None:
                rates.append(results[num_eps][seed]["final_success_rate"])
        
        if rates:
            avg = np.mean(rates)
            std = np.std(rates)
            print(f"  {num_eps} episodes: {avg:.4f} ± {std:.4f}")
        else:
            print(f"  {num_eps} episodes: No results")
    
    print()
    
    # Save summary
    summary_file = Path(__file__).parent.parent / "results_summary.json"
    summary_data = {
        "episode_sizes": episode_sizes,
        "seeds": seeds,
        "results": {
            str(num_eps): {
                str(seed): results[num_eps][seed]
                for seed in seeds
            }
            for num_eps in episode_sizes
        }
    }
    
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    summarize_results()