#!/usr/bin/env bash

set -u
set -o pipefail

# ============================================================
# Generic MetaWorld SmolVLA checkpoint sweep
#
# Usage:
#   bash personal/work2/eval_model/sweep_checkpoints_generic.sh \
#       <MODEL_ROOT> \
#       <CAMERA_NAME> \
#       <LABEL> \
#       [GPU_ID] \
#       [N_EPISODES] \
#       [BATCH_SIZE]
#
# Example:
#   bash personal/work2/eval_model/sweep_checkpoints_generic.sh \
#       personal/work2/duibi/random_42_corner3/random_112_seed42 \
#       corner3,gripperPOV \
#       random_corner3 \
#       1 50 4
#
# Output:
#   personal/work2/eval_model/checkpoint_sweep/<LABEL>/
#       eval_002000.log
#       ...
#       summary.csv
#       ranking.csv
#       best_checkpoint.txt
#
# Ranking:
#   1. pc_success
#   2. pc_grasp_success
#   3. avg_sum_reward
#
# Notes:
#   - uses --env.use_self_mw=true
#   - does NOT use rename_map
#   - intended as validation/screening sweep
# ============================================================

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

if [[ $# -lt 3 ]]; then
    echo "Usage:"
    echo "  bash $0 <MODEL_ROOT> <CAMERA_NAME> <LABEL> [GPU_ID] [N_EPISODES] [BATCH_SIZE]"
    exit 1
fi

MODEL_ROOT="$1"
CAMERA_NAME="$2"
LABEL="$3"
GPU_ID="${4:-1}"
N_EPISODES="${5:-50}"
BATCH_SIZE="${6:-4}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

OUTPUT_ROOT="personal/work2/eval_model/checkpoint_sweep/${LABEL}"

STEPS=(
    002000
    004000
    006000
    008000
    010000
    012000
    014000
    016000
)

mkdir -p "${OUTPUT_ROOT}"

SUMMARY_CSV="${OUTPUT_ROOT}/summary.csv"
echo "step,pc_success,pc_grasp_success,avg_sum_reward,avg_max_reward,n_episodes,status" > "${SUMMARY_CSV}"

echo "============================================================"
echo "Checkpoint sweep"
echo "Label:               ${LABEL}"
echo "Model root:          ${MODEL_ROOT}"
echo "Camera:              ${CAMERA_NAME}"
echo "GPU:                 ${GPU_ID}"
echo "Episodes/checkpoint: ${N_EPISODES}"
echo "Batch size:          ${BATCH_SIZE}"
echo "Output:              ${OUTPUT_ROOT}"
echo "============================================================"

for STEP in "${STEPS[@]}"; do
    CHECKPOINT="${MODEL_ROOT}/checkpoints/${STEP}/pretrained_model"
    LOG_FILE="${OUTPUT_ROOT}/eval_${STEP}.log"

    echo
    echo "============================================================"
    echo "Evaluating ${LABEL} checkpoint ${STEP}"
    echo "Path: ${CHECKPOINT}"
    echo "============================================================"

    if [[ ! -d "${CHECKPOINT}" ]]; then
        echo "WARNING: checkpoint does not exist: ${CHECKPOINT}"
        echo "${STEP},,,,,,missing" >> "${SUMMARY_CSV}"
        continue
    fi

    lerobot-eval \
        --policy.path="${CHECKPOINT}" \
        --env.type=metaworld \
        --env.task=pick-place-v3 \
        --env.camera_name="${CAMERA_NAME}" \
        --env.use_self_mw=true \
        --eval.batch_size="${BATCH_SIZE}" \
        --eval.n_episodes="${N_EPISODES}" \
        --policy.device=cuda \
        2>&1 | tee "${LOG_FILE}"

    EVAL_EXIT=${PIPESTATUS[0]}

    if [[ "${EVAL_EXIT}" -ne 0 ]]; then
        echo "ERROR: eval failed for checkpoint ${STEP}"
        echo "${STEP},,,,,,failed" >> "${SUMMARY_CSV}"
        continue
    fi

    PARSED=$(
        python - "${LOG_FILE}" <<'PY'
import ast
import sys

log_path = sys.argv[1]

metric = None

with open(log_path, "r", errors="ignore") as f:
    for line in f:
        if "pc_success" not in line:
            continue

        start = line.find("{")
        end = line.rfind("}")

        if start < 0 or end < start:
            continue

        raw = line[start:end + 1]

        try:
            value = ast.literal_eval(raw)
        except Exception:
            continue

        if isinstance(value, dict) and "pc_success" in value:
            metric = value

if metric is None:
    print("PARSE_FAILED")
    raise SystemExit(0)

keys = [
    "pc_success",
    "pc_grasp_success",
    "avg_sum_reward",
    "avg_max_reward",
    "n_episodes",
]

print(",".join(str(metric.get(k, "")) for k in keys))
PY
    )

    if [[ "${PARSED}" == "PARSE_FAILED" ]]; then
        echo "WARNING: could not parse final metrics for ${STEP}"
        echo "${STEP},,,,,,parse_failed" >> "${SUMMARY_CSV}"
        continue
    fi

    echo "${STEP},${PARSED},ok" >> "${SUMMARY_CSV}"

    echo
    echo "Checkpoint ${STEP}: ${PARSED}"
done

echo
echo "============================================================"
echo "Checkpoint sweep finished: ${LABEL}"
echo "============================================================"
echo
echo "Raw summary:"
column -s, -t "${SUMMARY_CSV}" 2>/dev/null || cat "${SUMMARY_CSV}"

python - "${SUMMARY_CSV}" "${OUTPUT_ROOT}" "${MODEL_ROOT}" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
output_root = Path(sys.argv[2])
model_root = Path(sys.argv[3])

rows = []

with csv_path.open(newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["status"] != "ok":
            continue

        try:
            row["pc_success"] = float(row["pc_success"])
            row["pc_grasp_success"] = float(row["pc_grasp_success"])
            row["avg_sum_reward"] = float(row["avg_sum_reward"])
            row["avg_max_reward"] = float(row["avg_max_reward"])
            row["n_episodes"] = int(float(row["n_episodes"]))
        except (TypeError, ValueError):
            continue

        rows.append(row)

if not rows:
    print("No valid checkpoint results found.")
    raise SystemExit(0)

rows.sort(
    key=lambda r: (
        r["pc_success"],
        r["pc_grasp_success"],
        r["avg_sum_reward"],
    ),
    reverse=True,
)

ranking_path = output_root / "ranking.csv"

with ranking_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "rank",
        "step",
        "pc_success",
        "pc_grasp_success",
        "avg_sum_reward",
        "avg_max_reward",
        "n_episodes",
    ])

    for rank, row in enumerate(rows, start=1):
        writer.writerow([
            rank,
            row["step"],
            row["pc_success"],
            row["pc_grasp_success"],
            row["avg_sum_reward"],
            row["avg_max_reward"],
            row["n_episodes"],
        ])

print()
print("============================================================")
print("CHECKPOINT RANKING")
print("============================================================")

for rank, row in enumerate(rows, start=1):
    successes = round(row["pc_success"] * row["n_episodes"] / 100.0)

    print(
        f"{rank:2d}. "
        f"step={row['step']} "
        f"success={row['pc_success']:.1f}% "
        f"({successes}/{row['n_episodes']}) "
        f"grasp={row['pc_grasp_success']:.1f}% "
        f"reward={row['avg_sum_reward']:.2f}"
    )

best = rows[0]
best_path = model_root / "checkpoints" / best["step"] / "pretrained_model"

with (output_root / "best_checkpoint.txt").open("w") as f:
    f.write(str(best_path) + "\n")

print()
print("============================================================")
print("BEST SCREENING CHECKPOINT")
print("============================================================")
print(f"step:          {best['step']}")
print(f"success:       {best['pc_success']:.1f}%")
print(f"grasp success: {best['pc_grasp_success']:.1f}%")
print(f"avg reward:    {best['avg_sum_reward']:.2f}")
print(f"checkpoint:    {best_path}")
print()
print(f"Summary:       {csv_path}")
print(f"Ranking:       {ranking_path}")
print(f"Best path:     {output_root / 'best_checkpoint.txt'}")
PY
