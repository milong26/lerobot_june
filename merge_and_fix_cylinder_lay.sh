#!/bin/bash

# Merge and fix cylinder_lay datasets
# This script performs two operations:
# 1. Merge cylinder_lay and cylinder_lay_old datasets
# 2. Fix the episode numbering in the merged dataset

set -e

echo "=========================================="
echo "Step 1: Merge datasets"
echo "=========================================="

lerobot-edit-dataset \
    --new_repo_id ep10/cylinder_lay \
    --operation.type merge \
    --operation.repo_ids "['ep10/cylinder_lay', 'ep10/cylinder_lay_old']"

echo ""
echo "=========================================="
echo "Step 2: Fix episode numbering"
echo "=========================================="

lerobot-fix-merged-dataset \
    --repo_id ep10/cylinder_lay \
    --root /home/qwe/.cache/huggingface/lerobot/ep10_old_mixed/cylinder_lay \
    --output_repo_id ep10/cylinder_lay_fixed \
    --output_root /home/qwe/.cache/huggingface/lerobot/ep10_old_mixed/cylinder_lay_fixed

echo ""
echo "=========================================="
echo "All operations completed successfully!"
echo "=========================================="
echo "Merged dataset: /home/qwe/.cache/huggingface/lerobot/ep10_old_mixed/cylinder_lay"
echo "Fixed dataset: /home/qwe/.cache/huggingface/lerobot/ep10_old_mixed/cylinder_lay_fixed"