#!/usr/bin/env bash

set -u
set -o pipefail


# ============================================================
# MetaWorld Checkpoint Sweep
#
# 直接运行：
#
#   bash personal/work2/eval_model/sweep_checkpoints_generic.sh
#
# 不需要任何命令行参数。
# ============================================================


# ============================================================
# 1. 配置
# ============================================================

# 模型根目录
MODEL_ROOT="personal/work2/duibi/ours_v4_112_seed42_corner/dynamicgrid_v4_112_seed42"

# 相机
CAMERA_NAME="corner,gripperPOV"

# 本次实验名称
LABEL="our_v4_corner"

# 物理 GPU 编号
GPU_ID=1

# 每个 checkpoint 测试 episode 数
N_EPISODES=200

# evaluation batch size
BATCH_SIZE=4

# 最大测试 step
# 例如：
#   12000 -> 最多测试到 12k
#   16000 -> 最多测试到 16k
MAX_STEP=12000


# ============================================================
# 2. 环境配置
# ============================================================

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="${GPU_ID}"


# ============================================================
# 3. 输出路径
# ============================================================

OUTPUT_ROOT="personal/work2/eval_model/sweep_cps/${LABEL}"

SUMMARY_CSV="${OUTPUT_ROOT}/summary.csv"
RANKING_CSV="${OUTPUT_ROOT}/ranking.csv"
BEST_CHECKPOINT_FILE="${OUTPUT_ROOT}/best_checkpoint.txt"
RUN_LOG="${OUTPUT_ROOT}/run.log"


# 创建输出目录
mkdir -p "${OUTPUT_ROOT}"


# ============================================================
# 4. 所有脚本自身输出写入 run.log
#
# 从这里开始：
#
#   echo
#   printf
#   Python print
#   Shell error
#
# 都进入 run.log
#
# 不再打印到终端。
# ============================================================

exec > "${RUN_LOG}" 2>&1


# ============================================================
# 5. 写入基本信息
# ============================================================

printf '%s\n' "============================================================"
printf '%s\n' "MetaWorld Checkpoint Sweep"
printf '%s\n' "============================================================"
printf 'Model root: %s\n' "${MODEL_ROOT}"
printf 'Camera: %s\n' "${CAMERA_NAME}"
printf 'Label: %s\n' "${LABEL}"
printf 'GPU: %s\n' "${GPU_ID}"
printf 'Episodes/checkpoint: %s\n' "${N_EPISODES}"
printf 'Batch size: %s\n' "${BATCH_SIZE}"
printf 'Max step: %s\n' "${MAX_STEP}"
printf 'Output: %s\n' "${OUTPUT_ROOT}"


# ============================================================
# 6. 检查模型目录
# ============================================================

if [[ ! -d "${MODEL_ROOT}" ]]; then
    printf '[ERROR] Model root does not exist: %s\n' "${MODEL_ROOT}"
    exit 1
fi


CHECKPOINT_ROOT="${MODEL_ROOT}/checkpoints"


if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
    printf '[ERROR] Checkpoint root does not exist: %s\n' "${CHECKPOINT_ROOT}"
    exit 1
fi


# ============================================================
# 7. 初始化 summary.csv
# ============================================================

printf '%s\n' \
"step,pc_success,pc_grasp_success,avg_sum_reward,avg_max_reward,n_episodes,status" \
> "${SUMMARY_CSV}"


# ============================================================
# 8. 自动寻找可测试 checkpoint
# ============================================================

STEPS=()


for STEP_DIR in "${CHECKPOINT_ROOT}"/*; do

    # 路径必须存在
    [[ -e "${STEP_DIR}" ]] || continue

    # 必须是目录
    [[ -d "${STEP_DIR}" ]] || continue


    STEP="$(basename "${STEP_DIR}")"


    # 只接受纯数字目录
    #
    # 例如：
    #   002000
    #   004000
    #   012000
    #
    # 排除：
    #   final
    #   latest
    #   tmp
    #
    [[ "${STEP}" =~ ^[0-9]+$ ]] || continue


    # 转成十进制数字
    STEP_NUM=$((10#${STEP}))


    # 超过 MAX_STEP 不测试
    if (( STEP_NUM > MAX_STEP )); then
        continue
    fi


    # 必须存在 pretrained_model
    CHECKPOINT="${STEP_DIR}/pretrained_model"


    if [[ ! -d "${CHECKPOINT}" ]]; then
        printf '[SKIP] step=%s reason=missing_pretrained_model\n' "${STEP}"
        continue
    fi


    STEPS+=("${STEP}")

done


# ============================================================
# 9. 没找到 checkpoint
# ============================================================

if [[ "${#STEPS[@]}" -eq 0 ]]; then
    printf '[ERROR] No valid checkpoints found under %s\n' "${CHECKPOINT_ROOT}"
    exit 1
fi


# ============================================================
# 10. checkpoint 数字排序
# ============================================================

mapfile -t STEPS < <(
    printf '%s\n' "${STEPS[@]}" | sort -n
)


printf 'Available checkpoints:'

for STEP in "${STEPS[@]}"; do
    printf ' %s' "${STEP}"
done

printf '\n'

printf 'Total checkpoints: %d\n' "${#STEPS[@]}"


# ============================================================
# 11. 逐个测试 checkpoint
# ============================================================

for STEP in "${STEPS[@]}"; do

    CHECKPOINT="${CHECKPOINT_ROOT}/${STEP}/pretrained_model"

    LOG_FILE="${OUTPUT_ROOT}/eval_${STEP}.log"


    printf '[START] step=%s checkpoint=%s\n' \
        "${STEP}" \
        "${CHECKPOINT}"


    # ========================================================
    # 运行 evaluation
    #
    # 注意：
    #
    # 不使用 tee。
    #
    # 所有 lerobot-eval 输出只写入：
    #
    #   eval_002000.log
    #   eval_004000.log
    #   ...
    #
    # 不写终端，也不写 run.log。
    # ========================================================

    lerobot-eval \
        --policy.path="${CHECKPOINT}" \
        --env.type=metaworld \
        --env.task=pick-place-v3 \
        --env.camera_name="${CAMERA_NAME}" \
        --env.use_self_mw=true \
        --eval.batch_size="${BATCH_SIZE}" \
        --eval.n_episodes="${N_EPISODES}" \
        --policy.device=cuda \
        > "${LOG_FILE}" 2>&1


    EVAL_EXIT=$?


    # ========================================================
    # 12. evaluation 失败
    # ========================================================

    if [[ "${EVAL_EXIT}" -ne 0 ]]; then

        printf '[FAILED] step=%s exit_code=%s log=%s\n' \
            "${STEP}" \
            "${EVAL_EXIT}" \
            "${LOG_FILE}"

        printf '%s\n' \
            "${STEP},,,,,,failed" \
            >> "${SUMMARY_CSV}"

        continue

    fi


    # ========================================================
    # 13. 从 eval log 提取最终指标
    #
    # 提取：
    #
    #   pc_success
    #   pc_grasp_success
    #   avg_sum_reward
    #   avg_max_reward
    #   n_episodes
    #
    # 如果 log 中出现多次 pc_success，
    # 使用最后一个合法指标 dict。
    # ========================================================

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


print(
    ",".join(
        str(metric.get(key, ""))
        for key in keys
    )
)
PY
    )


    # ========================================================
    # 14. 指标解析失败
    # ========================================================

    if [[ "${PARSED}" == "PARSE_FAILED" ]]; then

        printf '[PARSE_FAILED] step=%s log=%s\n' \
            "${STEP}" \
            "${LOG_FILE}"

        printf '%s\n' \
            "${STEP},,,,,,parse_failed" \
            >> "${SUMMARY_CSV}"

        continue

    fi


    # ========================================================
    # 15. 保存 checkpoint 指标
    # ========================================================

    printf '%s\n' \
        "${STEP},${PARSED},ok" \
        >> "${SUMMARY_CSV}"


    printf '[DONE] step=%s metrics=%s\n' \
        "${STEP}" \
        "${PARSED}"

done


# ============================================================
# 16. 生成 ranking.csv
#
# 排名优先级：
#
# 1. pc_success
# 2. pc_grasp_success
# 3. avg_sum_reward
#
# 并生成：
#
# best_checkpoint.txt
# ============================================================

python - \
    "${SUMMARY_CSV}" \
    "${OUTPUT_ROOT}" \
    "${MODEL_ROOT}" <<'PY'

import csv
import sys

from pathlib import Path


summary_path = Path(sys.argv[1])
output_root = Path(sys.argv[2])
model_root = Path(sys.argv[3])

ranking_path = output_root / "ranking.csv"
best_path_file = output_root / "best_checkpoint.txt"


rows = []


# ============================================================
# 读取 summary
# ============================================================

with summary_path.open(newline="") as f:

    reader = csv.DictReader(f)

    for row in reader:

        if row["status"] != "ok":
            continue

        try:

            row["pc_success"] = float(
                row["pc_success"]
            )

            row["pc_grasp_success"] = float(
                row["pc_grasp_success"]
            )

            row["avg_sum_reward"] = float(
                row["avg_sum_reward"]
            )

            row["avg_max_reward"] = float(
                row["avg_max_reward"]
            )

            row["n_episodes"] = int(
                float(row["n_episodes"])
            )

        except (TypeError, ValueError):
            continue


        rows.append(row)


# ============================================================
# 没有有效结果
# ============================================================

if not rows:

    print("[ERROR] No valid evaluation results found.")

    raise SystemExit(0)


# ============================================================
# 排名
# ============================================================

rows.sort(
    key=lambda row: (
        row["pc_success"],
        row["pc_grasp_success"],
        row["avg_sum_reward"],
    ),
    reverse=True,
)


# ============================================================
# 保存 ranking.csv
# ============================================================

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


# ============================================================
# 保存 best_checkpoint.txt
# ============================================================

best = rows[0]


best_checkpoint = (
    model_root
    / "checkpoints"
    / best["step"]
    / "pretrained_model"
)


with best_path_file.open("w") as f:
    f.write(str(best_checkpoint) + "\n")


# ============================================================
# 把简洁排名信息写入 run.log
# ============================================================

print("[RANKING]")

for rank, row in enumerate(rows, start=1):

    success_count = round(
        row["pc_success"]
        * row["n_episodes"]
        / 100.0
    )

    grasp_count = round(
        row["pc_grasp_success"]
        * row["n_episodes"]
        / 100.0
    )

    print(
        f"rank={rank} "
        f"step={row['step']} "
        f"success={row['pc_success']:.1f}% "
        f"({success_count}/{row['n_episodes']}) "
        f"grasp={row['pc_grasp_success']:.1f}% "
        f"({grasp_count}/{row['n_episodes']}) "
        f"reward={row['avg_sum_reward']:.2f}"
    )


print(
    "[BEST] "
    f"step={best['step']} "
    f"success={best['pc_success']:.1f}% "
    f"grasp={best['pc_grasp_success']:.1f}% "
    f"checkpoint={best_checkpoint}"
)

PY


# ============================================================
# 17. 完成
# ============================================================

printf '[FINISHED] summary=%s\n' "${SUMMARY_CSV}"
printf '[FINISHED] ranking=%s\n' "${RANKING_CSV}"
printf '[FINISHED] best_checkpoint=%s\n' "${BEST_CHECKPOINT_FILE}"
printf '[FINISHED] run_log=%s\n' "${RUN_LOG}"