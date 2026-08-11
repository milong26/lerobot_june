#!/usr/bin/env python
"""
Generate 9 individual training scripts for random baseline experiments.
Each script launches in tmux with conda environment activation.
"""
import os
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "jobs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPISODE_SIZES = [100, 200, 300]
SEEDS = [42, 142, 242]
AVAILABLE_GPUS = [0, 1]
CONDA_ENV = "lb_server"

def generate_scripts():
    """Generate individual training scripts for each experiment."""
    
    print("=" * 60)
    print("Generating individual training scripts")
    print("=" * 60)
    print()
    
    gpu_idx = 0
    script_paths = []
    
    for num_eps in EPISODE_SIZES:
        for seed in SEEDS:
            gpu_id = AVAILABLE_GPUS[gpu_idx % len(AVAILABLE_GPUS)]
            exp_name = f"random_{num_eps}_seed{seed}"
            script_name = f"run_{exp_name}.sh"
            script_path = OUTPUT_DIR / script_name
            
            tmux_session = f"random_{num_eps}_s{seed}"
            log_dir = SCRIPT_DIR.parent / "logs"
            pid_file = log_dir / f"{exp_name}.pid"
            time_file = log_dir / f"{exp_name}.time"
            
            # Generate script content
            script_content = f'''#!/bin/bash
# Experiment: {exp_name}
# GPU: {gpu_id}
# tmux session: {tmux_session}

set -e

EXP_NAME="{exp_name}"
TMUX_SESSION="{tmux_session}"
LOG_DIR="{log_dir}"
PID_FILE="$LOG_DIR/$EXP_NAME.pid"
TIME_FILE="$LOG_DIR/$EXP_NAME.time"
TRAIN_SCRIPT="{SCRIPT_DIR}/train_random.sh"
NUM_EPISODES={num_eps}
SEED={seed}
GPU_ID={gpu_id}

# Create log directory
mkdir -p "$LOG_DIR"

# Kill existing tmux session if exists
tmux kill-session -t $TMUX_SESSION 2>/dev/null || true

# Record start time
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')" > "$TIME_FILE"

# Create a temporary runner script
RUNNER_SCRIPT="$LOG_DIR/$EXP_NAME"_runner.sh
cat > "$RUNNER_SCRIPT" << RUNNER_EOF
#!/bin/bash

# Initialize and activate conda
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate {CONDA_ENV}

# Record PID
echo $$ > "$PID_FILE"
echo "PID: $$" >> "$TIME_FILE"

echo "========================================"
echo "Experiment: $EXP_NAME"
echo "GPU: $GPU_ID"
echo "PID: $$"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

# Run training
bash "$TRAIN_SCRIPT" $NUM_EPISODES $SEED $GPU_ID 2>&1 | tee -a "$LOG_DIR/$EXP_NAME.log"

# Record end time
echo "" >> "$TIME_FILE"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$TIME_FILE"
echo "Status: completed" >> "$TIME_FILE"

echo ""
echo "========================================"
echo "Experiment completed: $EXP_NAME"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# Keep tmux session open
exec bash
RUNNER_EOF

chmod +x "$RUNNER_SCRIPT"

# Launch in tmux
tmux new-session -d -s $TMUX_SESSION "bash $RUNNER_SCRIPT"

echo "Launched experiment: $EXP_NAME"
echo "tmux session: $TMUX_SESSION"
echo "PID file: $PID_FILE"
echo "Time file: $TIME_FILE"
echo ""
echo "Monitor with: tmux attach -t $TMUX_SESSION"
echo "Check logs: tail -f $LOG_DIR/$EXP_NAME.log"
'''
            
            with open(script_path, "w") as f:
                f.write(script_content)
            
            os.chmod(script_path, 0o755)
            script_paths.append((exp_name, script_path, gpu_id, tmux_session))
            
            print(f"  Generated: {script_name} (GPU {gpu_id}, tmux: {tmux_session})")
            
            gpu_idx += 1
    
    print()
    print("=" * 60)
    print(f"Generated {len(script_paths)} scripts in: {OUTPUT_DIR}")
    print("=" * 60)
    print()
    print("Execute manually:")
    for exp_name, script_path, gpu_id, tmux_session in script_paths:
        print(f"  bash {script_path}  # {exp_name} on GPU {gpu_id}")
    print()
    
    # Generate execution list
    list_file = SCRIPT_DIR.parent / "execution_list.txt"
    with open(list_file, "w") as f:
        f.write("Random Baseline Experiments - Execution List\n")
        f.write("=" * 80 + "\n\n")
        f.write("Status Legend: [ ] Pending  [R] Running  [✓] Done  [✗] Failed\n\n")
        f.write(f"{'Status':<8} {'Experiment':<30} {'GPU':<5} {'tmux Session':<20} {'Script'}\n")
        f.write("-" * 100 + "\n")
        for exp_name, script_path, gpu_id, tmux_session in script_paths:
            f.write(f"[ ]      {exp_name:<30} {gpu_id:<5} {tmux_session:<20} bash {script_path}\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("Commands:\n")
        f.write("  tmux ls                    # List all sessions\n")
        f.write("  tmux attach -t <session>   # Attach to session\n")
        f.write("  cat logs/<exp>.time        # Check start/end time\n")
        f.write("  cat logs/<exp>.pid         # Check PID\n")
        f.write("  tail -f logs/<exp>.log     # Follow logs\n")
    
    print(f"Execution list saved to: {list_file}")


if __name__ == "__main__":
    generate_scripts()