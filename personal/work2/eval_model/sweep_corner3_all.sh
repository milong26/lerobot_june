#!/usr/bin/env bash

set -u
set -e

GPU_ID="${1:-1}"
N_EPISODES="${2:-50}"
BATCH_SIZE="${3:-4}"

SWEEP="personal/work2/eval_model/sweep_checkpoints_generic.sh"

echo "============================================================"
echo "corner3: Random / Uniform / Ours checkpoint validation sweep"
echo "GPU=${GPU_ID}, episodes=${N_EPISODES}, batch=${BATCH_SIZE}"
echo "============================================================"

bash "${SWEEP}" \
    personal/work2/duibi/random_42_corner3/random_112_seed42 \
    corner3,gripperPOV \
    random_corner3 \
    "${GPU_ID}" "${N_EPISODES}" "${BATCH_SIZE}"

bash "${SWEEP}" \
    personal/work2/duibi/uniform_42_corner3/uniform_112_seed42 \
    corner3,gripperPOV \
    uniform_corner3 \
    "${GPU_ID}" "${N_EPISODES}" "${BATCH_SIZE}"

bash "${SWEEP}" \
    personal/work2/duibi/ours_112_seed42_corner3/dynamicanchor_112_seed42 \
    corner3,gripperPOV \
    ours_corner3 \
    "${GPU_ID}" "${N_EPISODES}" "${BATCH_SIZE}"

echo
echo "All corner3 checkpoint sweeps finished."
echo "Results:"
echo "  personal/work2/eval_model/checkpoint_sweep/random_corner3/"
echo "  personal/work2/eval_model/checkpoint_sweep/uniform_corner3/"
echo "  personal/work2/eval_model/checkpoint_sweep/ours_corner3/"
