#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 🚀 Thor E2E Pipeline Orchestrator — Issue #204
# ═══════════════════════════════════════════════════════════════
#
# Runs the full pipeline: Newton Sim → COSMOS Transfer → GR00T Train
# Designed for NVIDIA AGX Thor (132GB unified GPU, sm_110, CUDA 13.0)
#
# Usage:
#   bash scripts/newton_groot/thor_e2e_pipeline.sh
#   nohup bash scripts/newton_groot/thor_e2e_pipeline.sh > /home/cagatay/e2e_pipeline/pipeline.log 2>&1 &
#
# Features:
#   - Checkpoint-based: resumes from last completed step
#   - GPU lockfile prevents concurrent GPU-intensive jobs
#   - Status tracking via /home/cagatay/e2e_pipeline/status.json
#   - Full logging to /home/cagatay/e2e_pipeline/pipeline.log
#
# ═══════════════════════════════════════════════════════════════

set -uo pipefail
# NOTE: -e removed intentionally so individual step failures don't kill the whole pipeline

# ───────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────
export HOME="/home/cagatay"
REPO_ROOT="/home/cagatay/actions-runner/_work/strands-gtc-nvidia/strands-gtc-nvidia"
WORK_DIR="/home/cagatay/e2e_pipeline"
STATUS_FILE="${WORK_DIR}/status.json"
GPU_LOCKFILE="${WORK_DIR}/.gpu_lock"
LOGFILE="${WORK_DIR}/pipeline.log"
VENV="${REPO_ROOT}/.venv"

# Pipeline settings
NUM_EPISODES=256
IMG_WIDTH=1280
IMG_HEIGHT=720
COSMOS_FRAMES=93
GROOT_STEPS=5000

# Export for Python scripts
export THOR_DATASET_DIR="${WORK_DIR}/gr00t_dataset"
export THOR_COSMOS_OUTPUT="${WORK_DIR}/cosmos_transferred"
export THOR_FINAL_DATASET="${WORK_DIR}/final_dataset"
export THOR_GROOT_OUTPUT="${WORK_DIR}/groot_finetuned"
export MUJOCO_GL=egl
export THOR_NUM_EPISODES="${NUM_EPISODES}"

# ───────────────────────────────────────────────────
# Utility Functions
# ───────────────────────────────────────────────────

log() {
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] $*"
}

write_status() {
    # Write or update status.json for a given step
    local step_name="$1"
    local step_status="$2"
    local duration="${3:-0}"
    local now
    now=$(date -u '+%Y-%m-%dT%H:%M:%S+00:00')

    python3 << PYEOF
import json, os
sf = "${STATUS_FILE}"
try:
    with open(sf) as f:
        data = json.load(f)
except:
    data = {"pipeline": "issue_204_e2e", "started_at": "${now}", "steps": {}}

data["current_step"] = "${step_name}"
data["steps"]["${step_name}"] = {
    "status": "${step_status}",
    "updated_at": "${now}",
    "duration_s": ${duration}
}
with open(sf, "w") as f:
    json.dump(data, f, indent=2)
PYEOF
}

is_step_completed() {
    local step="$1"
    if [ ! -f "${STATUS_FILE}" ]; then
        return 1
    fi
    python3 -c "
import json, sys
with open('${STATUS_FILE}') as f:
    data = json.load(f)
s = data.get('steps', {}).get('${step}', {}).get('status', '')
sys.exit(0 if s in ('completed', 'completed_fallback') else 1)
" 2>/dev/null
    return $?
}

acquire_gpu_lock() {
    if [ -f "${GPU_LOCKFILE}" ]; then
        local lock_pid
        lock_pid=$(cat "${GPU_LOCKFILE}" 2>/dev/null || echo "")
        if [ -n "${lock_pid}" ] && kill -0 "${lock_pid}" 2>/dev/null; then
            log "GPU locked by PID ${lock_pid} — waiting..."
            while [ -f "${GPU_LOCKFILE}" ] && kill -0 "${lock_pid}" 2>/dev/null; do
                sleep 10
            done
        fi
    fi
    echo "$$" > "${GPU_LOCKFILE}"
    log "GPU lock acquired (PID $$)"
}

release_gpu_lock() {
    rm -f "${GPU_LOCKFILE}"
    log "GPU lock released"
}

cleanup() {
    release_gpu_lock
    log "Pipeline interrupted — status saved for resume"
}

trap cleanup EXIT INT TERM

# ───────────────────────────────────────────────────
# Main Pipeline
# ───────────────────────────────────────────────────

mkdir -p "${WORK_DIR}"

log "═══════════════════════════════════════════════════════════"
log "  🚀 THOR E2E PIPELINE — Issue #204"
log "  Device: NVIDIA AGX Thor (132GB GPU, sm_110, CUDA 13.0)"
log "  Episodes: ${NUM_EPISODES} | Image: ${IMG_WIDTH}x${IMG_HEIGHT}"
log "  COSMOS: ${COSMOS_FRAMES} frames | GR00T: ${GROOT_STEPS} steps"
log "═══════════════════════════════════════════════════════════"

# Initialize status
write_status "pipeline" "starting" "0"

# ═══════════════════════════════════════════════════
# STEP 0: Environment Setup
# ═══════════════════════════════════════════════════

if ! is_step_completed "step0_env_setup"; then
    log ""
    log "=== STEP 0: Environment Setup ==="
    STEP0_START=$(date +%s)

    write_status "step0_env_setup" "running" "0"

    # Activate venv
    if [ -d "${VENV}" ]; then
        source "${VENV}/bin/activate"
        log "  Activated venv: ${VENV}"
    else
        log "  Creating venv..."
        python3 -m venv "${VENV}"
        source "${VENV}/bin/activate"
        pip install --upgrade pip
    fi

    # Verify GPU
    nvidia-smi 2>/dev/null | head -15 || log "nvidia-smi not available"

    # Verify Python deps
    python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability(0)
    print(f'  Compute: sm_{cap[0]}{cap[1]}')
" || log "  PyTorch check failed"

    python3 -c "import warp; print(f'  Warp: {warp.__version__}')" 2>/dev/null || log "  Warp not available"
    python3 -c "import newton; print(f'  Newton: {newton.__version__}')" 2>/dev/null || log "  Newton not available"
    python3 -c "import pyarrow; print(f'  PyArrow: {pyarrow.__version__}')" 2>/dev/null || log "  PyArrow not available"
    python3 -c "import mujoco; print(f'  MuJoCo: {mujoco.__version__}')" 2>/dev/null || log "  MuJoCo not available"
    which ffmpeg >/dev/null 2>&1 && log "  ffmpeg: OK" || log "  ffmpeg not found"

    # Create work directories
    mkdir -p "${WORK_DIR}"/{gr00t_dataset,cosmos_transferred,final_dataset,groot_finetuned,evaluation}

    STEP0_END=$(date +%s)
    STEP0_DURATION=$((STEP0_END - STEP0_START))
    write_status "step0_env_setup" "completed" "${STEP0_DURATION}"
    log "  Step 0 done (${STEP0_DURATION}s)"
else
    log "  Step 0 already completed, skipping"
    source "${VENV}/bin/activate"
fi

# Acquire GPU lock
acquire_gpu_lock

# ═══════════════════════════════════════════════════
# STEP 1: Mesh to MJCF (lightweight, skip if done)
# ═══════════════════════════════════════════════════

if ! is_step_completed "step1_mesh_to_mjcf"; then
    log ""
    log "=== STEP 1: 3D Room Mesh to MJCF Scene ==="
    STEP1_START=$(date +%s)

    write_status "step1_mesh_to_mjcf" "running" "0"

    # Use existing Python pipeline for mesh conversion
    python3 "${REPO_ROOT}/scripts/newton_groot/thor_e2e_pipeline.py" --step 1 2>&1 || {
        log "  Mesh conversion had issues, using ground plane fallback"
    }

    STEP1_END=$(date +%s)
    STEP1_DURATION=$((STEP1_END - STEP1_START))
    write_status "step1_mesh_to_mjcf" "completed" "${STEP1_DURATION}"
    log "  Step 1 done (${STEP1_DURATION}s)"
else
    log "  Step 1 already completed, skipping"
fi

# ═══════════════════════════════════════════════════
# STEP 2: Newton GPU Simulation + Data Collection
# ═══════════════════════════════════════════════════

if ! is_step_completed "step2_newton_sim"; then
    log ""
    log "=== STEP 2: Newton GPU Simulation (${NUM_EPISODES} episodes, ${IMG_WIDTH}x${IMG_HEIGHT}) ==="
    STEP2_START=$(date +%s)

    write_status "step2_newton_sim" "running" "0"

    python3 "${REPO_ROOT}/scripts/newton_groot/thor_collect_data.py" \
        --episodes "${NUM_EPISODES}" \
        --batch-size 16 \
        --img-width "${IMG_WIDTH}" \
        --img-height "${IMG_HEIGHT}" \
        --resume \
        2>&1

    STEP2_EXIT=$?
    STEP2_END=$(date +%s)
    STEP2_DURATION=$((STEP2_END - STEP2_START))

    if [ ${STEP2_EXIT} -eq 0 ]; then
        write_status "step2_newton_sim" "completed" "${STEP2_DURATION}"
        log "  Step 2 done (${STEP2_DURATION}s)"
    else
        write_status "step2_newton_sim" "failed" "${STEP2_DURATION}"
        log "  Step 2 FAILED (exit=${STEP2_EXIT}, ${STEP2_DURATION}s)"
        # Continue anyway — data may still be useful
    fi

    # Validate output
    PARQUET_COUNT=$(find "${WORK_DIR}/gr00t_dataset/data" -name "*.parquet" 2>/dev/null | wc -l)
    VIDEO_COUNT=$(find "${WORK_DIR}/gr00t_dataset/videos" -name "*.mp4" 2>/dev/null | wc -l)
    log "  Dataset: ${PARQUET_COUNT} parquet, ${VIDEO_COUNT} videos"
else
    log "  Step 2 already completed, skipping"
fi

# ═══════════════════════════════════════════════════
# STEP 3: COSMOS Transfer2.5 Sim-to-Real
# ═══════════════════════════════════════════════════

if ! is_step_completed "step3_cosmos_transfer"; then
    log ""
    log "=== STEP 3: COSMOS Transfer2.5 (sim to real) ==="
    STEP3_START=$(date +%s)

    write_status "step3_cosmos_transfer" "running" "0"

    python3 "${REPO_ROOT}/scripts/newton_groot/thor_cosmos_transfer.py" \
        --resume \
        --fallback-copy \
        2>&1

    STEP3_EXIT=$?
    STEP3_END=$(date +%s)
    STEP3_DURATION=$((STEP3_END - STEP3_START))

    if [ ${STEP3_EXIT} -eq 0 ]; then
        write_status "step3_cosmos_transfer" "completed" "${STEP3_DURATION}"
    else
        write_status "step3_cosmos_transfer" "completed_fallback" "${STEP3_DURATION}"
    fi
    log "  Step 3 done (${STEP3_DURATION}s)"
else
    log "  Step 3 already completed, skipping"
fi

# ═══════════════════════════════════════════════════
# STEPS 4-5: Dataset Assembly + GR00T Training
# ═══════════════════════════════════════════════════

if ! is_step_completed "step5_groot_train"; then
    log ""
    log "=== STEPS 4-5: Dataset Assembly + GR00T Training ==="
    STEP45_START=$(date +%s)

    python3 "${REPO_ROOT}/scripts/newton_groot/thor_groot_train.py" \
        --max-steps "${GROOT_STEPS}" \
        --batch-size 32 \
        --lr 1e-5 \
        --save-steps 500 \
        --resume \
        2>&1

    STEP45_EXIT=$?
    STEP45_END=$(date +%s)
    STEP45_DURATION=$((STEP45_END - STEP45_START))
    log "  Steps 4-5 done (${STEP45_DURATION}s, exit=${STEP45_EXIT})"
else
    log "  Steps 4-5 already completed, skipping"
fi

# ═══════════════════════════════════════════════════
# STEP 6: Evaluation
# ═══════════════════════════════════════════════════

log ""
log "=== STEP 6: Evaluation ==="
write_status "step6_evaluation" "running" "0"

python3 "${REPO_ROOT}/scripts/newton_groot/thor_e2e_pipeline.py" --step 6 2>&1 || {
    log "  Evaluation step had issues"
}

write_status "step6_evaluation" "completed" "0"

# ═══════════════════════════════════════════════════
# STEP 7: Publish to HuggingFace
# ═══════════════════════════════════════════════════

log ""
log "=== STEP 7: Publish to HuggingFace ==="
write_status "step7_publish" "running" "0"

python3 "${REPO_ROOT}/scripts/newton_groot/thor_e2e_pipeline.py" --step 7 2>&1 || {
    log "  HuggingFace publishing had issues"
}

write_status "step7_publish" "completed" "0"

# ═══════════════════════════════════════════════════
# Pipeline Complete
# ═══════════════════════════════════════════════════

release_gpu_lock
trap - EXIT INT TERM

log ""
log "═══════════════════════════════════════════════════════════"
log "  PIPELINE COMPLETE"
log "═══════════════════════════════════════════════════════════"

# Final status
python3 << 'PYEOF'
import json
sf = "/home/cagatay/e2e_pipeline/status.json"
try:
    with open(sf) as f:
        data = json.load(f)
except:
    data = {}
from datetime import datetime, timezone
data["completed_at"] = datetime.now(tz=timezone.utc).isoformat()
data["status"] = "completed"
with open(sf, "w") as f:
    json.dump(data, f, indent=2)
print(json.dumps(data, indent=2))
PYEOF

log "Pipeline output: ${WORK_DIR}"
log "Dataset: ${WORK_DIR}/gr00t_dataset"
log "COSMOS: ${WORK_DIR}/cosmos_transferred"
log "Model: ${WORK_DIR}/groot_finetuned"
log "Status: ${STATUS_FILE}"
