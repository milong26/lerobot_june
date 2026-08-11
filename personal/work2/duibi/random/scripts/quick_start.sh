#!/bin/bash
# Quick start guide for random baseline experiments
# Run this to launch all experiments

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "Random Baseline Experiments"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Episode sizes: 100, 200, 300"
echo "  Seeds: 42, 142, 242"
echo "  Total experiments: 9"
echo "  GPUs: 0, 1 (2 at a time)"
echo "  Steps: 20000"
echo "  Batch size: 32"
echo ""
echo "Launching experiments..."
echo ""

python run_all_experiments.py

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Monitor status:"
echo "   cat ../execution_status.json"
echo ""
echo "2. Check running tmux sessions:"
echo "   tmux ls"
echo "   tmux attach -t random_100_s42"
echo ""
echo "3. Check logs:"
echo "   tail -f ../logs/random_100_seed42.log"
echo ""
echo "4. Collect results (after all experiments complete):"
echo "   python collect_results.py"
echo ""