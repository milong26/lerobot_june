#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/zhonglinye/jun/lerobot"
cd "${REPO_ROOT}"

GPU_ID=1
SCAN_ROOT="personal/work2/duibi"

# Shared output root and manifest for both GPUs
SHARED_OUTPUT_ROOT="personal/work2/eval_model/checkpoint_each_model/results"
TASK_MANIFEST="${SHARED_OUTPUT_ROOT}/tasks_manifest.json"

# Per-GPU output directories (for per-seed csv, logs, eval_results)
OUTPUT_ROOT="personal/work2/eval_model/checkpoint_each_model/results_gpu${GPU_ID}"
PID_FILE="${OUTPUT_ROOT}/.worker.pid"
PER_SEED_DIR="${OUTPUT_ROOT}/per_seed"
LOG_DIR="${OUTPUT_ROOT}/logs"
RUN_LOG="${OUTPUT_ROOT}/run.log"

# Shared summary directory and CSV (both GPUs write to the same location)
SUMMARY_DIR="${SHARED_OUTPUT_ROOT}/per_checkpoint"
SUMMARY_CSV="${SHARED_OUTPUT_ROOT}/summary.csv"

N_EPISODES=200
BATCH_SIZE=4
SEED=1000
CAMERA_NAME="corner,gripperPOV"

DISCOVER_SCRIPT="personal/work2/eval_model/checkpoint_each_model/discover_tasks.py"
PROCESS_SCRIPT="personal/work2/eval_model/checkpoint_each_model/process_results.py"
BUILD_SUMMARY_SCRIPT="personal/work2/eval_model/checkpoint_each_model/build_summary.py"

mkdir -p "${OUTPUT_ROOT}" "${PER_SEED_DIR}" "${LOG_DIR}"
mkdir -p "${SUMMARY_DIR}" "${SHARED_OUTPUT_ROOT}"

# ============================================================
# Persistence: auto-restart via setsid/nohup if not already running
# ============================================================
if [[ -f "${PID_FILE}" ]]; then
    OLD_PID=$(cat "${PID_FILE}" 2>/dev/null || true)
    if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "[INFO] Worker already running with PID=${OLD_PID}. Log: ${RUN_LOG}"
        echo "[INFO] To restart, remove ${PID_FILE} after the worker finishes."
        tail -f "${RUN_LOG}" 2>/dev/null &
        TAIL_PID=$!
        trap "kill ${TAIL_PID} 2>/dev/null" EXIT
        wait ${TAIL_PID} 2>/dev/null || true
        exit 0
    else
        rm -f "${PID_FILE}"
    fi
fi

# If we are NOT already in a setsid session, re-exec ourselves under setsid
if [[ "${_EVAL_WORKER_STARTED:-}" != "1" ]]; then
    export _EVAL_WORKER_STARTED=1
    exec setsid bash "$0" "$@" </dev/null >"${RUN_LOG}" 2>&1 &
    NEW_PID=$!
    echo "${NEW_PID}" > "${PID_FILE}"
    echo "[INFO] Started persistent worker with PID=${NEW_PID} on GPU=${GPU_ID}"
    echo "[INFO] Log file: ${RUN_LOG}"
    echo "[INFO] Waiting for worker to initialize..."
    sleep 3
    tail -f "${RUN_LOG}" 2>/dev/null &
    TAIL_PID=$!
    trap "kill ${TAIL_PID} 2>/dev/null" EXIT
    wait ${TAIL_PID} 2>/dev/null || true
    exit 0
fi

# ============================================================
# From here: actual worker logic (runs under setsid)
# ============================================================
exec >> "${RUN_LOG}" 2>&1

echo "============================================================"
echo "MetaWorld Checkpoint Sweep - GPU ${GPU_ID}"
echo "============================================================"
echo "Start time: $(date)"
echo "GPU: ${GPU_ID}"
echo "Episodes/checkpoint: ${N_EPISODES}"
echo "Batch size: ${BATCH_SIZE}"
echo "Seed: ${SEED}"
echo "Camera: ${CAMERA_NAME}"
echo "Output: ${OUTPUT_ROOT}"

# ============================================================
# GPU check: set CUDA_VISIBLE_DEVICES FIRST, then check
# ============================================================
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

ACTUAL_GPU_NAME=$(python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(torch.cuda.get_device_name(0))" 2>&1) || {
    echo "[FATAL] CUDA is not available. Aborting."
    exit 1
}
echo "[GPU] Using GPU ${GPU_ID} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}): ${ACTUAL_GPU_NAME}"

# ============================================================
# Discover tasks: shared manifest (only generate once)
# ============================================================
(
    flock -x 200
    if [[ ! -f "${TASK_MANIFEST}" ]] || [[ ! -s "${TASK_MANIFEST}" ]]; then
        echo "[DISCOVER] Generating shared task manifest at ${TASK_MANIFEST} ..."
        python "${DISCOVER_SCRIPT}" > "${TASK_MANIFEST}" 2>&1
    else
        echo "[DISCOVER] Using existing shared task manifest at ${TASK_MANIFEST}"
    fi
) 200>"${SHARED_OUTPUT_ROOT}/.manifest.lock"

TOTAL_TASKS=$(wc -l < "${TASK_MANIFEST}")
echo "[DISCOVER] Found ${TOTAL_TASKS} total model-checkpoint tasks"

# ============================================================
# Split tasks: GPU0 gets even indices, GPU1 gets odd indices
# ============================================================
MY_TASKS_FILE="${OUTPUT_ROOT}/.my_tasks.jsonl"
> "${MY_TASKS_FILE}"

IDX=0
while IFS= read -r line; do
    if (( IDX % 2 == GPU_ID % 2 )); then
        echo "${line}" >> "${MY_TASKS_FILE}"
    fi
    IDX=$((IDX + 1))
done < "${TASK_MANIFEST}"

MY_TASK_COUNT=$(wc -l < "${MY_TASKS_FILE}")
echo "[ASSIGN] GPU${GPU_ID} assigned ${MY_TASK_COUNT} tasks (odd-indexed from shared manifest)"

if [[ "${MY_TASK_COUNT}" -eq 0 ]]; then
    echo "[INFO] No tasks assigned to GPU${GPU_ID}. Exiting."
    rm -f "${PID_FILE}"
    exit 0
fi

# ============================================================
# Process tasks
# ============================================================
DONE_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

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

    echo "[START] model=${MODEL} checkpoint=${CHECKPOINT}"

    # ============================================================
    # Checkpoint resume: strict validation before skipping
    # ============================================================
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
except Exception as e:
    print('no')
" 2>/dev/null || echo "no")
        if [[ "${VALIDATION_OK}" == "yes" ]]; then
            SHOULD_SKIP=true
        fi
    fi

    if [[ "${SHOULD_SKIP}" == "true" ]]; then
        SR=$(python -c "import json; d=json.load(open('${PER_CP_JSON}')); print(f\"{d['success_count']}/{d['n_episodes']}={d['pc_success']:.1f}%\")" 2>/dev/null || echo "?")
        GR=$(python -c "import json; d=json.load(open('${PER_CP_JSON}')); print(f\"{d['grasp_success_count']}/{d['n_episodes']}={d['pc_grasp_success']:.1f}%\")" 2>/dev/null || echo "?")
        echo "[SKIP] model=${MODEL} checkpoint=${CHECKPOINT} already complete: success=${SR} grasp=${GR}"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    # ============================================================
    # Run evaluation
    # ============================================================
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
        echo "[FAILED] model=${MODEL} checkpoint=${CHECKPOINT} exit_code=${EVAL_EXIT}"
        python "${PROCESS_SCRIPT}" failed "${PER_CP_JSON}" "${MODEL}" "${CHECKPOINT}" "${GPU_ID}" "${EVAL_EXIT}" "${EVAL_LOG}" "${CP_PATH}" 2>/dev/null || true
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    # ============================================================
    # Find eval_episode_results.json
    # ============================================================
    RESULTS_JSON=$(find "${RESULTS_DIR}" -name "eval_episode_results.json" -type f 2>/dev/null | head -1)

    if [[ -z "${RESULTS_JSON}" ]]; then
        echo "[FAILED] model=${MODEL} checkpoint=${CHECKPOINT} no eval_episode_results.json found"
        python "${PROCESS_SCRIPT}" failed "${PER_CP_JSON}" "${MODEL}" "${CHECKPOINT}" "${GPU_ID}" "no_results" "${EVAL_LOG}" "${CP_PATH}" 2>/dev/null || true
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    # ============================================================
    # Process results
    # ============================================================
    if python "${PROCESS_SCRIPT}" ok "${RESULTS_JSON}" "${PER_SEED_CSV}" "${PER_CP_JSON}" "${MODEL}" "${CHECKPOINT}" "${GPU_ID}" "${CP_PATH}" "${EVAL_LOG}"; then
        SR=$(python -c "import json; d=json.load(open('${PER_CP_JSON}')); print(f\"{d['success_count']}/{d['n_episodes']}={d['pc_success']:.1f}%\")" 2>/dev/null || echo "?")
        GR=$(python -c "import json; d=json.load(open('${PER_CP_JSON}')); print(f\"{d['grasp_success_count']}/{d['n_episodes']}={d['pc_grasp_success']:.1f}%\")" 2>/dev/null || echo "?")
        echo "[DONE] model=${MODEL} checkpoint=${CHECKPOINT} success=${SR} grasp=${GR}"
        DONE_COUNT=$((DONE_COUNT + 1))
    else
        echo "[FAILED] model=${MODEL} checkpoint=${CHECKPOINT} result processing failed"
        python "${PROCESS_SCRIPT}" failed "${PER_CP_JSON}" "${MODEL}" "${CHECKPOINT}" "${GPU_ID}" "parse_failed" "${EVAL_LOG}" "${CP_PATH}" 2>/dev/null || true
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    # ============================================================
    # Rebuild summary with file lock
    # ============================================================
    (
        flock -x 200
        python "${BUILD_SUMMARY_SCRIPT}" "${SUMMARY_DIR}" "${SUMMARY_CSV}" 2>/dev/null || true
    ) 200>"${SHARED_OUTPUT_ROOT}/.summary.lock"

done < "${MY_TASKS_FILE}"

# ============================================================
# Final summary
# ============================================================
echo "============================================================"
echo "GPU${GPU_ID} FINISHED"
echo "============================================================"
echo "Done: ${DONE_COUNT}"
echo "Skipped: ${SKIP_COUNT}"
echo "Failed: ${FAIL_COUNT}"
echo "Total processed: $((DONE_COUNT + SKIP_COUNT + FAIL_COUNT))"
echo "Summary: ${SUMMARY_CSV}"
echo "End time: $(date)"

rm -f "${PID_FILE}"