"""
Force sensor data analysis tool for grasping tasks
Features:
- Load npz files from force_comparison directory
- Apply Butterworth low-pass filter to L2 norm data
- Generate plots: L2 norm comparison + Gripper state
- Use the same rollback logic as main_controller.py to mark rollback trigger points
- Condition A: Force sensor difference detection
- Condition B: Gripper decrease trend detection
- Both conditions A and B must be satisfied to trigger need_to_rollback
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque

# Use default font (English only)
plt.rcParams['axes.unicode_minus'] = False


# Data format: each step should have predicted_force_norm, actual_force_norm, actual_gripper, and predicted_gripper
def load_npz_files(force_dir: str = "./force_comparison", gripper_dir: str = "./gripper_comparison"):
    """
    Load all force and gripper data npz files
    
    Returns:
        all_data: List of data dictionaries, each containing:
            - file: filename
            - step_indices: step index array
            - actual_force_norms: actual force L2 norm array (N,)
            - predicted_force_norms: predicted force L2 norm array (N,)
            - predicted_gripper: predicted gripper position array (if exists)
            - actual_gripper: actual gripper position array (if exists)
            - config: runtime configuration dict (if exists)
    """
    force_dir = Path(force_dir)
    force_files = sorted(force_dir.glob("*force_data_*.npz"))
    
    if not force_files:
        raise FileNotFoundError(f"No *force_data_*.npz files found in {force_dir}")
    
    # Build gripper data file mapping (timestamp -> file path)
    gripper_files = {}
    gripper_dir = Path(gripper_dir)
    if gripper_dir.exists():
        for gfile in sorted(gripper_dir.glob("*gripper_data_*.npz")):
            # Extract timestamp from filename (e.g., gripper_data_20260813_184137.npz -> 20260813_184137)
            timestamp = gfile.name.replace("gripper_data_", "").replace(".npz", "")
            gripper_files[timestamp] = gfile
    
    # Store all loaded data
    all_data = []
    for npz_path in force_files:
        data = np.load(npz_path, allow_pickle=True)
        
        # Extract timestamp from filename (e.g., force_data_20260813_184137.npz -> 20260813_184137)
        timestamp = npz_path.name.replace("force_data_", "").replace(".npz", "")
        
        # Initialize data dict with core force sensor data
        loaded_data = {
            'file': npz_path.name,
            'step_indices': data['step_indices'],
            'actual_force_norms': data['actual_force_norms'],
            'predicted_force_norms': data['predicted_force_norms']
        }
        
        # Try to load corresponding gripper data file (matched by timestamp)
        if timestamp in gripper_files:
            gfile = gripper_files[timestamp]
            gdata = np.load(gfile, allow_pickle=True)
            loaded_data['predicted_gripper'] = gdata['predicted_gripper']
            loaded_data['actual_gripper'] = gdata['actual_gripper']
        
        # Try to read config params from force sensor file
        if 'config_params' in data:
            # Config params stored as string, need to parse
            config_str = str(data['config_params'][0])
            try:
                import ast
                # Safely parse string to dict using ast.literal_eval
                loaded_data['config'] = ast.literal_eval(config_str)
                print(f"  [INFO] Loaded config from {npz_path.name}: {loaded_data['config']}")
            except Exception as e:
                # Use empty dict on parse failure
                print(f"  [WARN] Failed to parse config: {e}")
                loaded_data['config'] = {}
        else:
            loaded_data['config'] = {}
        
        all_data.append(loaded_data)
    
    return all_data



def check_rollback_condition(
    step_idx: int,
    actual_force: float,
    predicted_force: float,
    predicted_force_history: deque,
    actual_force_history: deque,
    actual_gripper_history: deque,
    config: dict,
) -> tuple:
    """
    Check if rollback is needed (using filtered L2 norms from saved data)
    
    Condition A: Force sensor difference detection
        - A1: Predicted force L2 norm / history mean > threshold
        - A2: Predicted force / actual force (delay compensation) > threshold
    
    Condition B: Gripper detection (based on config mode)
        - use_gripper_stable_check=True: gripper is stable (no longer decreasing or changing very little)
        - use_gripper_stable_check=False: gripper decrease trend detection (old logic)
    
    Both conditions A and B must be satisfied to trigger need_to_rollback
    
    Note: Data is already filtered in main_controller.py, no need to filter again
    
    Returns:
        (need_rollback, condition_a, condition_b, predicted_norm_filtered, history_mean, force_ratio)
    """
    # Add to history (data is already filtered)
    actual_force_history.append(actual_force)
    predicted_force_history.append(predicted_force)
    
    # Insufficient history data
    if len(actual_force_history) < 3:
        return False, False, False, 0.0, 0.0, 0.0
    
    # Exclude initial phase
    min_start_steps = config.get('min_start_steps', 100)
    if step_idx < min_start_steps:
        return False, False, False, 0.0, 0.0, 0.0
    
    force_ratio_multiplier = config.get('force_ratio_multiplier', 5.0)
    force_delay_steps = config.get('force_delay_steps', 30)
    grasp_history_window = config.get('grasp_history_window', 50)
    gripper_stable_threshold = config.get('gripper_stable_threshold', 0.5)
    use_gripper_stable_check = config.get('use_gripper_stable_check', False)
    
    # ===== Condition A1: Predicted force significantly larger than history mean =====
    condition_a1 = False
    history_size = len(predicted_force_history)
    history_start = max(0, history_size - grasp_history_window - 1)
    history_predicted_norms = list(predicted_force_history)[history_start:-1]
    
    predicted_norm_filtered = predicted_force  # Data is already filtered
    history_mean = 0.0
    
    if len(history_predicted_norms) >= 10:
        # Data is already filtered, directly calculate mean
        history_array = np.array(history_predicted_norms)
        history_mean = np.mean(history_array)
        
        if history_mean > 1e-6:
            relative_ratio = predicted_norm_filtered / history_mean
            
            if relative_ratio > force_ratio_multiplier:
                condition_a1 = True
    
    # ===== Condition A2: Actual force / history predicted force (delay compensation) > threshold =====
    # Logic: actual force lags, compare current actual force with predicted force from force_delay_steps ago
    # If actual force is much smaller than historical prediction, the predicted force didn't occur, environment mismatch
    # Data is already filtered, use directly
    condition_a2 = False
    if len(predicted_force_history) > force_delay_steps:
        delayed_predicted_norm = list(predicted_force_history)[-force_delay_steps - 1]
    else:
        delayed_predicted_norm = predicted_force
    
    if actual_force < 1e-6:
        condition_a2 = delayed_predicted_norm > 1e-6
    else:
        condition_a2 = (delayed_predicted_norm / actual_force) > force_ratio_multiplier
    
    force_ratio = delayed_predicted_norm / actual_force if actual_force >= 1e-6 else float('inf')
    
    # Condition A: A1 or A2
    condition_a = condition_a1 or condition_a2
    
    # ===== Condition B: Gripper detection (based on config mode) =====
    condition_b = False
    if len(actual_gripper_history) >= 3:
        if use_gripper_stable_check:
            # New mode: detect if gripper is stabilizing
            if len(actual_gripper_history) >= 10:
                recent_values = list(actual_gripper_history)[-10:]
                diffs = [abs(recent_values[i+1] - recent_values[i]) for i in range(len(recent_values)-1)]
                avg_change = np.mean(diffs)
                
                if avg_change < gripper_stable_threshold:
                    condition_b = True
        else:
            # Old mode: detect gripper decrease trend
            history_values = list(actual_gripper_history)[:-1]
            current_value = actual_gripper_history[-1]
            
            if len(history_values) >= 10:
                history_mean_gripper = np.mean(history_values)
                
                if current_value < history_mean_gripper:
                    decrease_ratio = (history_mean_gripper - current_value) / (history_mean_gripper + 1e-6)
                    
                    if decrease_ratio > 0.1:
                        condition_b = True
    
    # Both conditions A and B must be satisfied to trigger rollback (consistent with main_controller.py)
    need_rollback = condition_a and condition_b
    
    return need_rollback, condition_a, condition_b, predicted_norm_filtered, history_mean, force_ratio



def analyze_all_data(data_list, save_dir="./force_analysis"):
    """
    Analyze all data files using the same rollback logic as main_controller.py
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    
    for data in data_list:
        filename = data['file'].replace('.npz', '')
        print(f"\n{'='*60}")
        print(f"[INFO] Analyzing file: {data['file']}")
        print(f"  - Data points: {len(data['step_indices'])}")

        # Read config params from npz file
        config = data.get('config', {})
        force_ratio_multiplier = config.get('force_ratio_multiplier', 5.0)
        min_start_steps = config.get('min_start_steps', 100)
        grasp_history_window = config.get('grasp_history_window', 50)
        force_delay_steps = config.get('force_delay_steps', 30)
        gripper_decrease_threshold = config.get('gripper_decrease_threshold', 10)
        gripper_stable_threshold = config.get('gripper_stable_threshold', 0.5)
        use_gripper_stable_check = config.get('use_gripper_stable_check', False)
        cutoff_freq = config.get('force_filter_cutoff_freq', 2.0)
        fs = config.get('force_sampling_rate', 30.0)
        # 配置，有一些是默认的
        print(f"  [INFO] Using runtime config:")
        print(f"    - force_ratio_multiplier={force_ratio_multiplier}")
        print(f"    - min_start_steps={min_start_steps}")
        print(f"    - grasp_history_window={grasp_history_window}")
        print(f"    - force_delay_steps={force_delay_steps}")
        print(f"    - gripper_decrease_threshold={gripper_decrease_threshold}")
        print(f"    - gripper_stable_threshold={gripper_stable_threshold}")
        print(f"    - use_gripper_stable_check={use_gripper_stable_check}")
        print(f"    - cutoff_freq={cutoff_freq}Hz, fs={fs}Hz")

        # Use L2 norm data (no longer using 15D raw data)
        actual_force = data['actual_force_norms']
        predicted_force = data['predicted_force_norms']
        step_indices = data['step_indices']
        predicted_gripper = data.get('predicted_gripper', None)
        actual_gripper = data.get('actual_gripper', None)

        # 窗口为什么要这么设置？？TODO:
        predicted_force_history = deque(maxlen=grasp_history_window * 2)
        actual_force_history = deque(maxlen=force_delay_steps + 50)
        actual_gripper_history = deque(maxlen=grasp_history_window * 2)
        
        # Frame-by-frame rollback detection
        rollback_indices = []
        condition_a_indices = []
        condition_b_indices = []
        predicted_norms_filtered = []
        history_means = []
        force_ratios = []
        
        # Consecutive failure detection (consistent with main_controller.py)
        max_consecutive_failures = config.get('max_consecutive_failures', 3)
        rollback_limited = 0  # Consecutive rollback detection count
        
        for i in range(len(step_indices)):
            # Save actual gripper history (one value per step)
            if actual_gripper is not None and len(actual_gripper) > i:
                actual_gripper_history.append(float(actual_gripper[i]))
            
            need_rollback, cond_a, cond_b, pred_norm_filt, hist_mean, f_ratio = check_rollback_condition(
                step_indices[i],
                actual_force[i],
                predicted_force[i],
                predicted_force_history,
                actual_force_history,
                actual_gripper_history,
                config,
            )
            
            # Update consecutive failure count (consistent with update_rollback_status in main_controller.py)
            if need_rollback:
                rollback_limited += 1
            else:
                rollback_limited = 0
            
            # Trigger rollback only when consecutive failures exceed threshold
            if rollback_limited >= max_consecutive_failures:
                rollback_indices.append(i)
            
            if cond_a:
                condition_a_indices.append(i)
            if cond_b:
                condition_b_indices.append(i)
            
            predicted_norms_filtered.append(pred_norm_filt)
            history_means.append(hist_mean)
            force_ratios.append(f_ratio)
        
        # Filter L2 norms (for plotting)
        
        print(f"  - Detected {len(rollback_indices)} rollback trigger points")
        print(f"    - Condition A triggered: {len(condition_a_indices)} times")
        print(f"    - Condition B triggered: {len(condition_b_indices)} times")
        if rollback_indices:
            print(f"  - Trigger time points: {step_indices[rollback_indices[:10]].tolist()}{'...' if len(rollback_indices) > 10 else ''}")
        
        # ========== Figure: L2 Norm Comparison + Gripper + Mark Rollback Trigger Points ==========
        fig2, axes2 = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
        
        # Top: Raw L2 Norm
        axes2[0].plot(step_indices, actual_force, 'b-', alpha=0.5, linewidth=1, label='Actual (raw)')
        axes2[0].plot(step_indices, predicted_force, 'r-', alpha=0.5, linewidth=1, label='Predicted (raw)')
        axes2[0].set_ylabel('Force L2 Norm')
        axes2[0].set_title('Raw L2 Norm')
        axes2[0].legend()
        axes2[0].grid(True, alpha=0.3)
        
        # Second: Filtered L2 Norm
        valid_predicted_filtered = [v if v > 0 else predicted_force[i] for i, v in enumerate(predicted_norms_filtered)]
        axes2[1].plot(step_indices, actual_force, 'b-', alpha=0.7, linewidth=1.5, label='Actual (filtered)')
        axes2[1].plot(step_indices, valid_predicted_filtered, 'r-', alpha=0.7, linewidth=1.5, label='Predicted (filtered)')
        axes2[1].plot(step_indices, history_means, 'g--', alpha=0.5, linewidth=1, label='History Mean')
        
        # Mark rollback trigger points
        if rollback_indices:
            for rb_step in step_indices[rollback_indices]:
                axes2[1].axvline(x=rb_step, color='red', linestyle='-', linewidth=1.5, alpha=0.5, zorder=2)
        
        axes2[1].set_ylabel('Force L2 Norm (Filtered)')
        axes2[1].set_title(f'Filtered L2 Norm (Rollbacks: {len(rollback_indices)}, A:{len(condition_a_indices)}, B:{len(condition_b_indices)})')
        axes2[1].legend(fontsize=8, loc='best')
        axes2[1].grid(True, alpha=0.3)
        
        # Third: Predicted/Actual Ratio
        valid_ratios = np.where(np.isfinite(force_ratios), force_ratios, 0)
        axes2[2].plot(step_indices, valid_ratios, 'g-', linewidth=1.5, label='Predicted/Actual Ratio')
        axes2[2].axhline(y=force_ratio_multiplier, color='r', linestyle='--', linewidth=1, 
                        label=f'Threshold ({force_ratio_multiplier}x)')
        
        axes2[2].set_ylabel('Force Ratio')
        axes2[2].set_title('Predicted / Actual Force Ratio')
        axes2[2].legend()
        axes2[2].grid(True, alpha=0.3)
        
        # Bottom: Gripper State
        if predicted_gripper is not None and len(predicted_gripper) > 0:
            all_gripper_values = []
            all_gripper_steps = []
            for step_idx, g in enumerate(predicted_gripper):
                if isinstance(g, np.ndarray):
                    values = g.tolist()
                else:
                    values = [float(g)]
                all_gripper_values.extend(values)
                all_gripper_steps.extend([step_idx + j/len(values) for j in range(len(values))])
            
            axes2[3].plot(all_gripper_steps, all_gripper_values, 'm-', linewidth=1, alpha=0.7, label='Predicted Gripper')
            
            if actual_gripper is not None and len(actual_gripper) > 0:
                # 对齐数据长度：取 step_indices 和 actual_gripper 的最小长度
                min_len = min(len(step_indices), len(actual_gripper))
                gripper_steps = step_indices[:min_len]
                gripper_values = actual_gripper[:min_len]
                axes2[3].plot(gripper_steps, gripper_values, 'c-', linewidth=1.5, alpha=0.8, label='Actual Gripper')
            
            if rollback_indices:
                for rb_step in step_indices[rollback_indices]:
                    axes2[3].axvline(x=rb_step, color='red', linestyle='-', linewidth=1.5, alpha=0.5, zorder=2)
            
            axes2[3].set_ylabel('Gripper Value')
            axes2[3].set_title('Gripper State')
            axes2[3].set_xlabel('Step Index')
            axes2[3].legend(fontsize=8, loc='best')
            axes2[3].grid(True, alpha=0.3)
        else:
            axes2[3].text(0.5, 0.5, 'No Gripper Data', ha='center', va='center', transform=axes2[3].transAxes)
            axes2[3].set_xlabel('Step Index')
        
        plt.tight_layout()
        output_path2 = save_dir / f"{filename}_L2_norm_filtered.png"
        plt.savefig(output_path2, dpi=150, bbox_inches='tight')
        print(f"  - L2 norm plot saved: {output_path2}")
        plt.close()
        
        # Save filtered data
        npz_output = save_dir / f"{filename}_all_filtered.npz"
        np.savez_compressed(
            npz_output,
            step_indices=step_indices,
            actual_norm_raw=actual_force,
            predicted_norm_raw=predicted_force,
            rollback_indices=np.array(rollback_indices),
            condition_a_indices=np.array(condition_a_indices),
            condition_b_indices=np.array(condition_b_indices),
            force_ratios=np.array(force_ratios),
        )
        print(f"  - Filtered data saved: {npz_output}")
        


def main():
    """Main function"""
    data_dir = "./force_comparison"
    gripper_dir = "./gripper_comparison"
    save_dir = "./force_analysis"
    # 返回若干个dict，一个timestamp对应一个dict
    data_list = load_npz_files(data_dir, gripper_dir)
    
    if not data_list:
        print("[ERROR] No data files found, exiting")
        return
    
    analyze_all_data(data_list, save_dir=save_dir)
    
if __name__ == "__main__":
    main()