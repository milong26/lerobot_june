#!/usr/bin/env python
"""
Run all random baseline experiments with GPU management.
Uses GPUs 0 and 1, runs 2 experiments at a time, maintains execution status.
"""
import json
import os
import subprocess
import time
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent
LEROBOT_ROOT = SCRIPT_DIR.parent.parent.parent.parent

EPISODE_SIZES = [100, 200, 300]
SEEDS = [42, 142, 242]
AVAILABLE_GPUS = [0, 1]  # Only use GPUs 0 and 1
STATUS_FILE = SCRIPT_DIR.parent / "execution_status.json"


def load_status():
    """Load execution status from file."""
    if STATUS_FILE.exists():
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    return {"experiments": {}}


def save_status(status):
    """Save execution status to file."""
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)


def init_status():
    """Initialize status for all experiments."""
    status = load_status()
    
    for num_eps in EPISODE_SIZES:
        for seed in SEEDS:
            exp_key = f"{num_eps}_{seed}"
            if exp_key not in status["experiments"]:
                status["experiments"][exp_key] = {
                    "num_episodes": num_eps,
                    "seed": seed,
                    "status": "pending",  # pending, running, completed, failed
                    "gpu_id": None,
                    "tmux_session": None,
                    "start_time": None,
                    "end_time": None,
                }
    
    save_status(status)
    return status


def get_running_sessions():
    """Get list of currently running tmux sessions."""
    try:
        result = subprocess.run(
            ["tmux", "list-sessions"],
            capture_output=True,
            text=True,
            check=True
        )
        sessions = []
        for line in result.stdout.strip().split("\n"):
            if line:
                session_name = line.split(":")[0]
                sessions.append(session_name)
        return sessions
    except subprocess.CalledProcessError:
        return []


def is_session_running(session_name):
    """Check if a tmux session is running."""
    sessions = get_running_sessions()
    return session_name in sessions


def print_status(status):
    """Print current execution status."""
    print("=" * 80)
    print("Random Baseline Experiments - Execution Status")
    print("=" * 80)
    print()
    
    pending = 0
    running = 0
    completed = 0
    failed = 0
    
    for num_eps in EPISODE_SIZES:
        print(f"Episodes: {num_eps}")
        for seed in SEEDS:
            exp_key = f"{num_eps}_{seed}"
            exp = status["experiments"][exp_key]
            exp_status = exp["status"]
            gpu = exp["gpu_id"] if exp["gpu_id"] is not None else "-"
            
            status_icon = {
                "pending": "○",
                "running": "●",
                "completed": "✓",
                "failed": "✗"
            }.get(exp_status, "?")
            
            print(f"  {status_icon} seed={seed:3d} | GPU: {gpu} | Status: {exp_status}")
            
            if exp_status == "pending":
                pending += 1
            elif exp_status == "running":
                running += 1
            elif exp_status == "completed":
                completed += 1
            elif exp_status == "failed":
                failed += 1
        print()
    
    print("-" * 80)
    print(f"Summary: {pending} pending, {running} running, {completed} completed, {failed} failed")
    print("=" * 80)


def launch_experiment(exp_key, exp_data, gpu_id):
    """Launch a single experiment."""
    num_eps = exp_data["num_episodes"]
    seed = exp_data["seed"]
    session_name = f"random_{num_eps}_s{seed}"
    train_script = SCRIPT_DIR / "train_random.sh"
    
    # Kill existing session if it exists
    subprocess.run(
        ["tmux", "kill-session", "-t", session_name],
        capture_output=True
    )
    
    # Build command
    cmd = f"bash {train_script} {num_eps} {seed} {gpu_id}"
    
    print(f"  Launching: {session_name} on GPU {gpu_id}")
    
    # Create tmux session
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_name, cmd],
        check=True
    )
    
    return session_name


def main():
    # Initialize status
    status = init_status()
    
    print("Starting Random Baseline Experiments")
    print(f"Available GPUs: {AVAILABLE_GPUS}")
    print(f"Total experiments: {len(EPISODE_SIZES) * len(SEEDS)}")
    print()
    
    # Main loop
    while True:
        # Refresh status
        status = load_status()
        
        # Check for completed/failed experiments
        for exp_key, exp_data in status["experiments"].items():
            if exp_data["status"] == "running" and exp_data["tmux_session"]:
                if not is_session_running(exp_data["tmux_session"]):
                    # Check if it completed successfully
                    num_eps = exp_data["num_episodes"]
                    seed = exp_data["seed"]
                    log_file = SCRIPT_DIR.parent / "logs" / f"random_{num_eps}_seed{seed}.log"
                    
                    if log_file.exists():
                        with open(log_file, "r") as f:
                            content = f.read()
                            if "Training complete" in content:
                                exp_data["status"] = "completed"
                                print(f"  ✓ Completed: {exp_key}")
                            else:
                                exp_data["status"] = "failed"
                                print(f"  ✗ Failed: {exp_key}")
                    else:
                        exp_data["status"] = "failed"
                        print(f"  ✗ Failed (no log): {exp_key}")
                    
                    exp_data["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    exp_data["gpu_id"] = None
                    exp_data["tmux_session"] = None
                    save_status(status)
        
        # Count running experiments
        running_count = sum(
            1 for exp in status["experiments"].values()
            if exp["status"] == "running"
        )
        
        # Get available GPUs
        used_gpus = set()
        for exp in status["experiments"].values():
            if exp["status"] == "running" and exp["gpu_id"] is not None:
                used_gpus.add(exp["gpu_id"])
        
        available_gpus = [gpu for gpu in AVAILABLE_GPUS if gpu not in used_gpus]
        
        # Launch new experiments if there's capacity
        if available_gpus:
            # Find pending experiments
            pending_exps = [
                (key, exp) for key, exp in status["experiments"].items()
                if exp["status"] == "pending"
            ]
            
            if pending_exps and available_gpus:
                # Launch up to len(available_gpus) experiments
                for i, (exp_key, exp_data) in enumerate(pending_exps):
                    if i >= len(available_gpus):
                        break
                    
                    gpu_id = available_gpus[i]
                    session_name = launch_experiment(exp_key, exp_data, gpu_id)
                    
                    exp_data["status"] = "running"
                    exp_data["gpu_id"] = gpu_id
                    exp_data["tmux_session"] = session_name
                    exp_data["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    save_status(status)
        
        # Print status
        print_status(status)
        
        # Check if all done
        all_done = all(
            exp["status"] in ["completed", "failed"]
            for exp in status["experiments"].values()
        )
        
        if all_done:
            print("\nAll experiments completed!")
            break
        
        # Wait before next check
        time.sleep(30)


if __name__ == "__main__":
    main()