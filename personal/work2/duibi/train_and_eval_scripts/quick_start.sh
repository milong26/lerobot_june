#!/bin/bash
# Quick start guide for uniform and random experiments
# Run this to launch all experiments

# chmod +x /data/zhonglinye/jun/lerobot/personal/work2/duibi/train_and_eval_scripts/*.sh

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "Uniform & Random Baseline Experiments"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Episode count: 100"
echo "  Uniform seed: 42"
echo "  Random seeds: 42, 100"
echo "  Total experiments: 3"
echo "  Steps: 20000"
echo "  Batch size: 64"
echo ""
echo "Launching experiments..."
echo ""

# Launch uniform experiment
echo "[1/3] Launching uniform_100_seed42..."
bash launch_uniform.sh 0
echo ""

# Launch random experiments
echo "[2/3] Launching random_100_seed42..."
bash launch_random.sh 42 0
echo ""

echo "[3/3] Launching random_100_seed100..."
bash launch_random.sh 100 0
echo ""

echo "=========================================="
echo "All experiments launched!"
echo "=========================================="
echo ""
echo "Next Steps:"
echo ""
echo "1. Check running tmux sessions:"
echo "   tmux ls"
echo ""
echo "2. Attach to sessions:"
echo "   tmux attach -t uniform_100_s42"
echo "   tmux attach -t random_100_s42"
echo "   tmux attach -t random_100_s100"
echo ""
echo "3. Check logs:"
echo "   tail -f /data/zhonglinye/jun/lerobot/personal/work2/duibi/uniform_42/logs/uniform_100_seed42.log"
echo "   tail -f /data/zhonglinye/jun/lerobot/personal/work2/duibi/random_42/logs/random_100_seed42.log"
echo "   tail -f /data/zhonglinye/jun/lerobot/personal/work2/duibi/random_100/logs/random_100_seed100.log"
echo ""
echo "4. Check results:"
echo "   cat /data/zhonglinye/jun/lerobot/personal/work2/duibi/uniform_42/eval_results/uniform_100_seed42_eval.json"
echo "   cat /data/zhonglinye/jun/lerobot/personal/work2/duibi/random_42/eval_results/random_100_seed42_eval.json"
echo "   cat /data/zhonglinye/jun/lerobot/personal/work2/duibi/random_100/eval_results/random_100_seed100_eval.json"
echo ""