"""
Force sensor data analysis tool for grasping tasks (新版状态机)
Features:
- Load npz files from force_comparison directory
- 可视化状态机轨迹（Gripper State Machine）
- 可视化力传感器数据（原始值 + 因果滤波后）
- 使用 filtfilt 进行离线双向滤波（仅用于可视化对比）
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt

# Use default font (English only)
plt.rcParams['axes.unicode_minus'] = False


def load_npz_files(force_dir: str = "./force_comparison", gripper_dir: str = "./gripper_comparison"):
    """
    Load all force and gripper data npz files (新版格式)
    
    Returns:
        all_data: List of data dictionaries, each containing:
            - file: filename
            - step_indices: step index array
            - gripper_states: gripper state machine states
            - trajectory_steps: state machine trajectory steps
            - trajectory_states: state machine trajectory states
            - trajectory_actual_norms: trajectory actual norms (因果滤波后)
            - trajectory_predicted_norms: trajectory predicted norms (因果滤波后)
            - predicted_gripper: predicted gripper position array
            - actual_gripper: actual gripper position array
            - config: runtime configuration dict
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
            timestamp = gfile.name.replace("gripper_data_", "").replace(".npz", "")
            gripper_files[timestamp] = gfile
    
    all_data = []
    for npz_path in force_files:
        data = np.load(npz_path, allow_pickle=True)
        timestamp = npz_path.name.replace("force_data_", "").replace(".npz", "")
        
        loaded_data = {
            'file': npz_path.name,
            'step_indices': data['step_indices'],
            'gripper_states': data['gripper_states'],
        }
        
        # 加载状态机轨迹（因果滤波后的力范数）
        if 'trajectory_steps' in data:
            loaded_data['trajectory_steps'] = data['trajectory_steps']
            loaded_data['trajectory_states'] = data['trajectory_states']
            loaded_data['trajectory_actual_norms'] = data['trajectory_actual_norms']
            loaded_data['trajectory_predicted_norms'] = data['trajectory_predicted_norms']
        
        # Try to load corresponding gripper data file
        if timestamp in gripper_files:
            gfile = gripper_files[timestamp]
            gdata = np.load(gfile, allow_pickle=True)
            loaded_data['predicted_gripper'] = gdata['predicted_gripper']
            loaded_data['actual_gripper'] = gdata['actual_gripper']
        
        # Try to read config params
        if 'config_params' in data:
            config_str = str(data['config_params'][0])
            try:
                import ast
                loaded_data['config'] = ast.literal_eval(config_str)
                print(f"  [INFO] Loaded config from {npz_path.name}")
            except Exception as e:
                print(f"  [WARN] Failed to parse config: {e}")
                loaded_data['config'] = {}
        else:
            loaded_data['config'] = {}
        
        all_data.append(loaded_data)
    
    return all_data


def butterworth_lowpass_1d(signal, cutoff_freq=2.0, fs=30.0, order=4):
    """Butterworth 低通滤波（1D信号，仅用于离线可视化）"""
    min_length = 3 * (order + 1) + 1
    if len(signal) <= min_length:
        return signal
    
    nyquist = fs / 2.0
    normalized_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)


def analyze_all_data(data_list, save_dir="./force_analysis"):
    """
    Analyze all data files with state machine visualization and phase interval marking
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Phase colors for background shading
    phase_colors = {
        'closing': '#FFE4B5',        # 浅橙色
        'stable_candidate': '#E0FFFF', # 浅青色
        'grasp_confirmed': '#90EE90',  # 浅绿色
        'failed_no_grasp': '#FFB6C1',  # 浅红色
    }
    
    for data in data_list:
        filename = data['file'].replace('.npz', '')
        print(f"\n{'='*60}")
        print(f"[INFO] Analyzing file: {data['file']}")
        print(f"  - Data points: {len(data['step_indices'])}")
        
        config = data.get('config', {})
        
        # 打印配置
        print(f"  [INFO] Config:")
        for key in ['min_start_steps', 'force_filter_cutoff_freq', 'force_sampling_rate',
                    'gripper_velocity_threshold', 'stable_window', 'settle_steps', 
                    'sustain_steps', 'max_closing_duration', 'force_ratio_threshold',
                    'gripper_close_threshold', 'initial_gripper_value']:
            if key in config:
                print(f"    - {key}={config[key]}")
        
        # 基础数据
        step_indices = data['step_indices']
        gripper_states = data['gripper_states']
        predicted_gripper = data.get('predicted_gripper', None)
        actual_gripper = data.get('actual_gripper', None)
        
        # 状态机轨迹（因果滤波后的力范数）
        trajectory_steps = data.get('trajectory_steps', None)
        trajectory_states = data.get('trajectory_states', None)
        trajectory_actual = data.get('trajectory_actual_norms', None)
        trajectory_predicted = data.get('trajectory_predicted_norms', None)
        
        # 使用轨迹数据（因果滤波后）作为 filtered 数据
        if trajectory_actual is not None and len(trajectory_actual) > 0:
            actual_force_filtered = trajectory_actual
            predicted_force_filtered = trajectory_predicted
            filtered_steps = trajectory_steps
        else:
            actual_force_filtered = np.array([])
            predicted_force_filtered = np.array([])
            filtered_steps = step_indices
        
        print(f"  - Gripper states: {np.unique(gripper_states)}")
        if trajectory_steps is not None:
            print(f"  - Trajectory points: {len(trajectory_steps)}")
        
        # ========== 识别阶段区间 ==========
        phase_intervals = []
        if trajectory_states is not None and len(trajectory_states) > 0:
            current_state = trajectory_states[0]
            start_idx = 0
            
            for i in range(1, len(trajectory_states)):
                if trajectory_states[i] != current_state:
                    phase_intervals.append({
                        'state': current_state,
                        'start': trajectory_steps[start_idx],
                        'end': trajectory_steps[i-1],
                        'start_idx': start_idx,
                        'end_idx': i-1
                    })
                    current_state = trajectory_states[i]
                    start_idx = i
            
            # 添加最后一个区间
            phase_intervals.append({
                'state': current_state,
                'start': trajectory_steps[start_idx],
                'end': trajectory_steps[-1],
                'start_idx': start_idx,
                'end_idx': len(trajectory_states)-1
            })
            
            print(f"  - Phase intervals:")
            for interval in phase_intervals:
                print(f"    {interval['state']}: step {interval['start']} - {interval['end']}")
        
        # ========== 可视化 ==========
        fig, axes = plt.subplots(4, 1, figsize=(18, 13), sharex=True)
        
        # 绘制阶段区间背景
        for interval in phase_intervals:
            color = phase_colors.get(interval['state'], '#FFFFFF')
            for ax in axes:
                ax.axvspan(interval['start'], interval['end'], 
                          alpha=0.3, color=color, 
                          label=f"{interval['state']}")
        
        # Top: Filtered L2 Norm (因果滤波后)
        axes[0].plot(filtered_steps, actual_force_filtered, 'b-', alpha=0.8, linewidth=1.5, label='Actual (causal filtered)')
        axes[0].plot(filtered_steps, predicted_force_filtered, 'r-', alpha=0.8, linewidth=1.5, label='Predicted (causal filtered)')
        axes[0].set_ylabel('Force L2 Norm')
        axes[0].set_title('Force L2 Norm (Causal Filtered)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Second: Force Ratio (Actual / Predicted)
        force_ratio_threshold = config.get('force_ratio_threshold', 0.5)
        
        # 计算比值
        safe_predicted = np.where(np.abs(predicted_force_filtered) > 1e-6, 
                                  predicted_force_filtered, 1e-6)
        force_ratio = actual_force_filtered / safe_predicted
        
        axes[1].plot(filtered_steps, force_ratio, 'm-', alpha=0.7, linewidth=1.5, label='Force Ratio (actual/predicted)')
        axes[1].axhline(y=force_ratio_threshold, color='r', linestyle='--', linewidth=2, 
                       label=f'Ratio Threshold ({force_ratio_threshold})')
        axes[1].axhline(y=1.0, color='g', linestyle=':', linewidth=1, label='Perfect Match (1.0)')
        axes[1].set_ylabel('Force Ratio')
        axes[1].set_title('Force Ratio (Actual / Predicted)')
        axes[1].legend(fontsize=8, loc='best')
        axes[1].grid(True, alpha=0.3)
        
        # Third: Gripper State Machine Trajectory
        state_map = {
            'closing': 0,
            'stable_candidate': 1,
            'grasp_confirmed': 2,
            'failed_no_grasp': -1,
        }
        
        if trajectory_steps is not None and len(trajectory_steps) > 0:
            state_values = [state_map.get(s, 0) for s in trajectory_states]
            
            axes[2].plot(trajectory_steps, state_values, 'm-', linewidth=2, label='Gripper State')
            axes[2].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
            axes[2].axhline(y=1, color='gray', linestyle=':', alpha=0.5)
            axes[2].axhline(y=2, color='gray', linestyle=':', alpha=0.5)
            
            axes[2].set_yticks([-1, 0, 1, 2])
            axes[2].set_yticklabels(['FAILED_NO_GRASP', 'CLOSING', 
                                    'STABLE_CANDIDATE', 'GRASP_CONFIRMED'])
            
            # 标记失败区域
            fail_mask = np.array(state_values) < 0
            if np.any(fail_mask):
                fail_steps = trajectory_steps[fail_mask]
                for fs in fail_steps:
                    axes[2].axvline(x=fs, color='red', linestyle='-', linewidth=1, alpha=0.3)
            
            axes[2].set_ylabel('Gripper State')
            axes[2].set_title('Gripper State Machine Trajectory')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'No State Machine Trajectory', 
                       ha='center', va='center', transform=axes[2].transAxes)
        
        # Fourth: Gripper Position
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
            
            axes[3].plot(all_gripper_steps, all_gripper_values, 'm-', linewidth=1, alpha=0.7, label='Predicted Gripper')
            
            if actual_gripper is not None and len(actual_gripper) > 0:
                min_len = min(len(step_indices), len(actual_gripper))
                gripper_steps = step_indices[:min_len]
                gripper_values = actual_gripper[:min_len]
                axes[3].plot(gripper_steps, gripper_values, 'c-', linewidth=1.5, alpha=0.8, label='Actual Gripper')
            
            # 显示 initial_gripper_value 和 gripper_close_threshold
            initial_gripper = config.get('initial_gripper_value', None)
            close_threshold = config.get('gripper_close_threshold', None)
            
            if initial_gripper is not None:
                axes[3].axhline(y=initial_gripper, color='blue', linestyle='--', linewidth=1, 
                               alpha=0.5, label=f'Initial Gripper ({initial_gripper:.2f})')
            
            if close_threshold is not None:
                # 显示闭合检测阈值线（initial_gripper - threshold）
                close_threshold_value = initial_gripper - close_threshold if initial_gripper is not None else close_threshold
                axes[3].axhline(y=close_threshold_value, color='orange', linestyle='--', linewidth=1, 
                               alpha=0.5, label=f'Close Threshold ({close_threshold_value:.2f})')
            
            axes[3].set_ylabel('Gripper Value')
            axes[3].set_title('Gripper Position')
            axes[3].set_xlabel('Step Index')
            axes[3].legend(fontsize=8, loc='best')
            axes[3].grid(True, alpha=0.3)
        else:
            axes[3].text(0.5, 0.5, 'No Gripper Data', ha='center', va='center', transform=axes[3].transAxes)
            axes[3].set_xlabel('Step Index')
        
        # 添加图例说明阶段颜色
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, alpha=0.3, label=state) 
                          for state, color in phase_colors.items()]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
                  fontsize=8, title='Phase Intervals')
        
        plt.tight_layout()
        output_path = save_dir / f"{filename}_state_machine.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  - State machine plot saved: {output_path}")
        plt.close()


def main():
    """Main function"""
    data_dir = "./force_comparison"
    gripper_dir = "./gripper_comparison"
    save_dir = "./force_analysis"
    
    data_list = load_npz_files(data_dir, gripper_dir)
    
    if not data_list:
        print("[ERROR] No data files found, exiting")
        return
    
    analyze_all_data(data_list, save_dir=save_dir)
    

if __name__ == "__main__":
    main()