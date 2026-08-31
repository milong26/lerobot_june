#!/usr/bin/env bash

set -u

# ============================================================
# Sweep all checkpoints for:
# Random + corner3
#
# Screening metric:
#   1. pc_success
#   2. pc_grasp_success
#   3. avg_sum_reward
#
# This script DOES NOT use rename_map.
# ============================================================

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

GPU_ID=${1:-1}
N_EPISODES=${2:-50}
BATCH_SIZE=${3:-4}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

MODEL_ROOT="personal/work2/duibi/random_42_corner3/random_112_seed42"
OUTPUT_ROOT="personal/work2/eval_model/checkpoint_sweep/random_corner3"

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

echo "step,pc_success,pc_grasp_success,avg_sum_reward,avg_max_reward,n_episodes,status" \
    > "${SUMMARY_CSV}"

echo "============================================================"
echo "Random + corner3 checkpoint sweep"
echo "GPU: ${GPU_ID}"
echo "Episodes/checkpoint: ${N_EPISODES}"
echo "Batch size: ${BATCH_SIZE}"
echo "Model root: ${MODEL_ROOT}"
echo "Output: ${OUTPUT_ROOT}"
echo "============================================================"

for STEP in "${STEPS[@]}"; do

    CHECKPOINT="${MODEL_ROOT}/checkpoints/${STEP}/pretrained_model"
    LOG_FILE="${OUTPUT_ROOT}/eval_${STEP}.log"

    echo
    echo "============================================================"
    echo "Evaluating checkpoint ${STEP}"
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
        --env.camera_name=corner3,gripperPOV \
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

    # --------------------------------------------------------
    # Parse final Overall Aggregated Metrics
    # --------------------------------------------------------

    PARSED=$(
        python - "${LOG_FILE}" <<'PY'
import re
import sys

log_path = sys.argv[1]

with open(log_path, "r", errors="ignore") as f:
    text = f.read()

# Find all metric dictionaries containing pc_success.
matches = re.findall(
    r"\{[^{}\n]*['\"]pc_success['\"]\s*:\s*[^{}\n]+\}",
    text
)

if not matches:
    print("PARSE_FAILED")
    sys.exit(0)

# Use the final aggregated metrics dictionary.
metric = matches[-1]

def get_number(key):
    pattern = rf"['\"]{re.escape(key)}['\"]\s*:\s*([-+0-9.eE]+)"
    m = re.search(pattern, metric)
    return m.group(1) if m else ""

values = [
    get_number("pc_success"),
    get_number("pc_grasp_success"),
    get_number("avg_sum_reward"),
    get_number("avg_max_reward"),
    get_number("n_episodes"),
]

print(",".join(values))
PY
    )

    if [[ "${PARSED}" == "PARSE_FAILED" ]]; then
        echo "WARNING: could not parse final metrics for ${STEP}"
        echo "${STEP},,,,,,parse_failed" >> "${SUMMARY_CSV}"
        continue
    fi

    echo "${STEP},${PARSED},ok" >> "${SUMMARY_CSV}"

    echo
    echo "Checkpoint ${STEP} result:"
    echo "${PARSED}"

done

echo
echo "============================================================"
echo "Checkpoint sweep finished"
echo "============================================================"
echo
echo "Raw summary:"
column -s, -t "${SUMMARY_CSV}" 2>/dev/null || cat "${SUMMARY_CSV}"

# ============================================================
# Rank checkpoints by success rate
# ============================================================

python - "${SUMMARY_CSV}" "${OUTPUT_ROOT}" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
output_root = Path(sys.argv[2])

rows = []

with csv_path.open() as f:
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
        except (ValueError, TypeError):
            continue

        rows.append(row)

if not rows:
    print("\nNo valid checkpoint results found.")
    sys.exit(0)

# Main criterion:
# 1. pc_success
# 2. pc_grasp_success
# 3. avg_sum_reward
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

print("\n============================================================")
print("CHECKPOINT RANKING")
print("============================================================")

for rank, row in enumerate(rows, start=1):
    print(
        f"{rank:2d}. step={row['step']} "
        f"success={row['pc_success']:.1f}% "
        f"grasp={row['pc_grasp_success']:.1f}% "
        f"reward={row['avg_sum_reward']:.2f}"
    )

best = rows[0]

print("\n============================================================")
print("BEST SCREENING CHECKPOINT")
print("============================================================")
print(f"step:          {best['step']}")
print(f"success:       {best['pc_success']:.1f}%")
print(f"grasp success: {best['pc_grasp_success']:.1f}%")
print(f"avg reward:    {best['avg_sum_reward']:.2f}")

best_path = (
    "personal/work2/duibi/random_42_corner3/"
    "random_112_seed42/checkpoints/"
    f"{best['step']}/pretrained_model"
)

print(f"checkpoint:    {best_path}")

with (output_root / "best_checkpoint.txt").open("w") as f:
    f.write(best_path + "\n")

print(f"\nRanking saved to: {ranking_path}")
print(f"Best path saved to: {output_root / 'best_checkpoint.txt'}")
PY