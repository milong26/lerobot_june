#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/zhonglinye/jun/lerobot"
cd "${REPO_ROOT}"

GPU_ID=1
SCRIPT_LABEL="partB"
TMUX_SESSION="eval_gpu1_partB"
SCAN_ROOT="personal/work2/duibi"

OUTPUT_ROOT="personal/work2/eval_model/checkpoint_each_model/results_gpu1_${SCRIPT_LABEL}"
PID_FILE="${OUTPUT_ROOT}/.worker.pid"
PER_SEED_DIR="${OUTPUT_ROOT}/per_seed"
LOG_DIR="${OUTPUT_ROOT}/logs"
RUN_LOG="${OUTPUT_ROOT}/run.log"

SUMMARY_DIR="${OUTPUT_ROOT}/per_checkpoint"
SUMMARY_CSV="${OUTPUT_ROOT}/summary.csv"

N_EPISODES=200
BATCH_SIZE=4
SEED=1000
CAMERA_NAME="corner,gripperPOV"

PROCESS_SCRIPT="personal/work2/eval_model/checkpoint_each_model/process_results.py"
BUILD_SUMMARY_SCRIPT="personal/work2/eval_model/checkpoint_each_model/build_summary.py"

MODELS_PART_B=(
    "random_42_corner/random_112_seed42"
    "random_42_corner2/random_112_seed42"
    "random_42_corner3/random_112_seed42"
    "subzerocore_112_seed42_corner/subzerocore_112_seed42"
    "uniform_42_corner/uniform_112_seed42"
    "uniform_42_corner2/uniform_112_seed42"
    "uniform_42_corner3/uniform_112_seed42"
)

mkdir -p "${OUTPUT_ROOT}" "${PER_SEED_DIR}" "${LOG_DIR}" "${SUMMARY_DIR}"

# ============================================================
# tmux persistence: if not running in tmux, relaunch in tmux
# ============================================================
if [[ -z "${TMUX:-}" ]] && [[ "${_EVAL_IN_TMUX:-}" != "1" ]]; then
    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
        echo "[INFO] tmux session '${TMUX_SESSION}' already exists."
        echo "[INFO] Attach with: tmux attach -t ${TMUX_SESSION}"
        echo "[INFO] View log: tail -f ${REPO_ROOT}/${RUN_LOG}"
        exit 0
    fi
    export _EVAL_IN_TMUX=1
    echo "[INFO] Launching eval worker in tmux session '${TMUX_SESSION}'..."
    tmux new-session -d -s "${TMUX_SESSION}" -c "${REPO_ROOT}" \
        "bash \"$0\" \"\$@\""
    sleep 2
    echo "[INFO] tmux session '${TMUX_SESSION}' started."
    echo "[INFO] Attach with: tmux attach -t ${TMUX_SESSION}"
    echo "[INFO] View log: tail -f ${REPO_ROOT}/${RUN_LOG}"
    exit 0
fi

# ============================================================
# From here: runs inside tmux session
# ============================================================
exec >> "${RUN_LOG}" 2>&1

echo "============================================================"
echo "MetaWorld Checkpoint Sweep - GPU ${GPU_ID} (Part B)"
echo "============================================================"
echo "Start time: $(date)"
echo "GPU: ${GPU_ID}"
echo "Episodes/checkpoint: ${N_EPISODES}"
echo "Batch size: ${BATCH_SIZE}"
echo "Seed: ${SEED}"
echo "Camera: ${CAMERA_NAME}"
echo "Output: ${OUTPUT_ROOT}"
echo "Models: ${#MODELS_PART_B[@]}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

ACTUAL_GPU_NAME=$(python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(torch.cuda.get_device_name(0))" 2>&1) || {
    echo "[FATAL] CUDA is not available. Aborting."
    exit 1
}
echo "[GPU] Using GPU ${GPU_ID} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}): ${ACTUAL_GPU_NAME}"

echo "[DISCOVER] Scanning checkpoints for assigned models..."

TASKS_FILE="${OUTPUT_ROOT}/.tasks.jsonl"
> "${TASKS_FILE}"

for MODEL_REL in "${MODELS_PART_B[@]}"; do
    MODEL_DIR="${SCAN_ROOT}/${MODEL_REL}"
    CP_ROOT="${MODEL_DIR}/checkpoints"
    if [[ ! -d "${CP_ROOT}" ]]; then
        echo "[WARN] No checkpoints dir for ${MODEL_REL}, skipping"
        continue
    fi
    for STEP_DIR in "${CP_ROOT}"/*/; do
        [[ -d "${STEP_DIR}" ]] || continue
        STEP_NAME=$(basename "${STEP_DIR}")
        [[ "${STEP_NAME}" =~ ^[0-9]+$ ]] || continue
        PRETRAINED="${STEP_DIR}/pretrained_model"
        if [[ ! -d "${PRETRAINED}" ]]; then
            echo "[SKIP] ${MODEL_REL}/${STEP_NAME}: no pretrained_model"
            continue
        fi
        STEP_NUM=$((10#${STEP_NAME}))
        printf '{"model":"%s","checkpoint":"%s","step_num":%d,"checkpoint_path":"%s"}\n' \
            "${MODEL_REL}" "${STEP_NAME}" "${STEP_NUM}" "${PRETRAINED}" >> "${TASKS_FILE}"
    done
done

python -c "
import json, sys
tasks = []
with open('${TASKS_FILE}') as f:
    for line in f:
        line = line.strip()
        if line:
            tasks.append(json.loads(line))
tasks.sort(key=lambda t: (t['model'], t['step_num']))
with open('${TASKS_FILE}', 'w') as f:
    for t in tasks:
        f.write(json.dumps(t) + '\n')
"

TOTAL_TASKS=$(wc -l < "${TASKS_FILE}")
echo "[DISCOVER] Found ${TOTAL_TASKS} tasks for Part B"

if [[ "${TOTAL_TASKS}" -eq 0 ]]; then
    echo "[INFO] No tasks for Part B. Exiting."
    rm -f "${PID_FILE}"
    exit 0
fi

echo "[TASKS] Assigned models and checkpoints:"
while IFS= read -r line; do
    M=$(echo "${line}" | python -c "import sys,json; print(json.loads(sys.stdin.read())['model'])")
    C=$(echo "${line}" | python -c "import sys,json; print(json.loads(sys.stdin.read())['checkpoint'])")
    echo "  - ${M} @ ${C}"
done < "${TASKS_FILE}"

DONE_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0
TOTAL_PROCESSED=0

while IFS= read -r task_line; do
    MODEL=$(echo "${task_line}" | python -c "import sys,json; print(json.loads(sys.stdin.read())['model'])")
    CHECKPOINT=$(echo "${task_line}" | python -c "import sys,json; print(json.loads(sys.stdin.read())['checkpoint'])")
    CP_PATH=$(echo "${task_line}" | python -c "import sys,json; print(json.loads(sys.stdin.read())['checkpoint_path'])")

    SAFE_MODEL=$(echo "${MODEL}" | sed 's/[^a-zA-Z0-9._-]/_/g')
    SAFE_CP=$(echo "${CHECKPOINT}" | sed 's/[^a-zA-Z0-9._-]/_/g')
    TASK_KEY="${SAFE_MODEL}__${SAFE_CP}"

    PER_SEED_CSV="${PER_SEED_DIR}/${TASK_KEY}.csv"
    PER_CP_JSON="${SUMMARY_DIR}/${TASK_KEY}.json"
    EVAL_LOG="${LOG_DIR}/eval_${TASK_KEY}.log"
    RESULTS_DIR="${OUTPUT_ROOT}/eval_results/${TASK_KEY}"

    TOTAL_PROCESSED=$((TOTAL_PROCESSED + 1))
    echo ""
    echo "============================================================"
    echo "[${TOTAL_PROCESSED}/${TOTAL_TASKS}] START model=${MODEL} checkpoint=${CHECKPOINT}"
    echo "============================================================"

    SHOULD_SKIP=false
    if [[ -f "${PER_CP_JSON}" ]] && [[ -f "${PER_SEED_CSV}" ]]; then
        VALIDATION_OK=$(python -c "
import csv, json, sys
try:
    with open('${PER_CP_JSON}') as f:
        summary = json.load(f)
    if summary.get('status') != 'ok':
        print('no')
        sys.exit(0)
    rows = []
    with open('${PER_SEED_CSV}') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if len(rows) != ${N_EPISODES}:
        print('no')
        sys.exit(0)
    seeds = set()
    for row in rows:
        seeds.add(int(row['seed']))
    if len(seeds) != ${N_EPISODES}:
        print('no')
        sys.exit(0)
    expected_seeds = set(range(${SEED}, ${SEED} + ${N_EPISODES}))
    if seeds != expected_seeds:
        print('no')
        sys.exit(0)
    success_count = sum(1 for r in rows if int(r['success']) == 1)
    grasp_count = sum(1 for r in rows if int(r['grasp_success']) == 1)
    if success_count != summary.get('success_count'):
        print('no')
        sys.exit(0)
    if grasp_count != summary.get('grasp_success_count'):
        print('no')
        sys.exit(0)
    print('yes')
except Exception:
    print('no')
" 2>/dev/null || echo "no")
        if [[ "${VALIDATION_OK}" == "yes" ]]; then
            SHOULD_SKIP=true
        fi
    fi

    if [[ "${SHOULD_SKIP}" == "true" ]]; then
        SR=$(python -c "import json; d=json.load(open('${PER_CP_JSON}')); print(f\"{d['success_count']}/{d['n_episodes']}={d['pc_success']:.1f}%\")" 2>/dev/null || echo "?")
        GR=$(python -c "import json; d=json.load(open('${PER_CP_JSON}')); print(f\"{d['grasp_success_count']}/{d['n_episodes']}={d['pc_grasp_success']:.1f}%\")" 2>/dev/null || echo "?")
        echo "[SKIP] checkpoint=${CHECKPOINT} already complete: success=${SR} grasp=${GR}"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    set +e
    lerobot-eval \
        --policy.path="${CP_PATH}" \
        --env.type=metaworld \
        --env.task=pick-place-v3 \
        --env.camera_name="${CAMERA_NAME}" \
        --env.use_self_mw=true \
        --eval.n_episodes="${N_EPISODES}" \
        --eval.batch_size="${BATCH_SIZE}" \
        --policy.device=cuda \
        --seed="${SEED}" \
        --output_dir="${RESULTS_DIR}" \
        > "${EVAL_LOG}" 2>&1
    EVAL_EXIT=$?
    set -e

    if [[ "${EVAL_EXIT}" -ne 0 ]]; then
        echo "[FAILED] checkpoint=${CHECKPOINT} exit_code=${EVAL_EXIT}"
        python "${PROCESS_SCRIPT}" failed "${PER_CP_JSON}" "${MODEL}" "${CHECKPOINT}" "${GPU_ID}" "${EVAL_EXIT}" "${EVAL_LOG}" "${CP_PATH}" 2>/dev/null || true
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    RESULTS_JSON=$(find "${RESULTS_DIR}" -name "eval_episode_results.json" -type f 2>/dev/null | head -1)

    if [[ -z "${RESULTS_JSON}" ]]; then
        echo "[FAILED] checkpoint=${CHECKPOINT} no eval_episode_results.json found"
        python "${PROCESS_SCRIPT}" failed "${PER_CP_JSON}" "${MODEL}" "${CHECKPOINT}" "${GPU_ID}" "no_results" "${EVAL_LOG}" "${CP_PATH}" 2>/dev/null || true
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    if python "${PROCESS_SCRIPT}" ok "${RESULTS_JSON}" "${PER_SEED_CSV}" "${PER_CP_JSON}" "${MODEL}" "${CHECKPOINT}" "${GPU_ID}" "${CP_PATH}" "${EVAL_LOG}"; then
        SR=$(python -c "import json; d=json.load(open('${PER_CP_JSON}')); print(f\"{d['success_count']}/{d['n_episodes']}={d['pc_success']:.1f}%\")" 2>/dev/null || echo "?")
        GR=$(python -c "import json; d=json.load(open('${PER_CP_JSON}')); print(f\"{d['grasp_success_count']}/{d['n_episodes']}={d['pc_grasp_success']:.1f}%\")" 2>/dev/null || echo "?")
        echo "[DONE] checkpoint=${CHECKPOINT} success=${SR} grasp=${GR}"
        DONE_COUNT=$((DONE_COUNT + 1))
    else
        echo "[FAILED] checkpoint=${CHECKPOINT} result processing failed"
        python "${PROCESS_SCRIPT}" failed "${PER_CP_JSON}" "${MODEL}" "${CHECKPOINT}" "${GPU_ID}" "parse_failed" "${EVAL_LOG}" "${CP_PATH}" 2>/dev/null || true
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    (
        flock -x 200
        python "${BUILD_SUMMARY_SCRIPT}" "${SUMMARY_DIR}" "${SUMMARY_CSV}" 2>/dev/null || true
    ) 200>"${OUTPUT_ROOT}/.summary.lock"

done < "${TASKS_FILE}"

echo "============================================================"
echo "GPU${GPU_ID} PartB FINISHED"
echo "============================================================"
echo "Done: ${DONE_COUNT}"
echo "Skipped: ${SKIP_COUNT}"
echo "Failed: ${FAIL_COUNT}"
echo "Total processed: $((DONE_COUNT + SKIP_COUNT + FAIL_COUNT))"
echo "Summary: ${SUMMARY_CSV}"
echo "End time: $(date)"

rm -f "${PID_FILE}"