#!/usr/bin/env python3
"""
Debug script to save evaluation data for analysis.
This script runs a few episodes and saves:
1. Input observations (images, state)
2. Language instructions
3. Predicted actions
4. Environment rewards and success flags
5. Action statistics
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

# Add lerobot to path
sys.path.insert(0, "/data/zhonglinye/jun/lerobot/src")

from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.envs.metaworld import create_metaworld_envs
from lerobot.processor.pipeline import PolicyProcessorPipeline


def main():
    # Configuration
    checkpoint_path = "/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/experiments/tinyvla_s_disassemble-v3_corner/checkpoints/tinyvla_tinyvla_s_disassemble-v3_corner/checkpoints/012000/pretrained_model"
    output_dir = Path("/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/debug_eval_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_episodes = 16
    batch_size = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {output_dir}")
    
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    
    # Create rename map
    rename_map = {
        "observation.images.camera1": "observation.images.top",
        "observation.images.camera2": "observation.images.wrist",
    }
    
    # Create environment
    print("\nCreating environment...")
    
    envs = create_metaworld_envs(
        task="disassemble-v3",
        n_envs=batch_size,
        gym_kwargs={
            "camera_name": "corner,gripperPOV",
            "use_self_mw": True,
            "obs_type": "pixels_agent_pos",
        },
        env_cls=gym.vector.AsyncVectorEnv,
    )
    env = envs["disassemble-v3"][0]
    print(f"Environment created: {env.num_envs} parallel envs")
    
    # Create policy
    print("\nCreating policy...")
    from lerobot.policies.tinyvla.configuration_tinyvla import TinyVLAConfig
    from lerobot.configs.policies import PreTrainedConfig
    
    # Load config from checkpoint
    config = PreTrainedConfig.from_pretrained(checkpoint_path)
    config.device = device
    
    # Create policy config
    from dataclasses import dataclass
    from types import SimpleNamespace
    
    policy_cfg = SimpleNamespace(
        type="tinyvla_s",
        device=device,
        pretrained_path=Path(checkpoint_path),
        lora_enable=True,
    )
    
    # Create env config for make_policy
    env_cfg = SimpleNamespace(
        type="metaworld",
        task="disassemble-v3",
        camera_name="corner,gripperPOV",
        use_self_mw=True,
        obs_type="pixels_agent_pos",
    )
    
    policy = make_policy(
        cfg=policy_cfg,
        env_cfg=env_cfg,
        rename_map=rename_map,
    )
    policy.eval()
    
    # Create preprocessors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=Path(checkpoint_path),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    
    print(f"Policy created: {type(policy).__name__}")
    print(f"Policy device: {policy.config.device}")
    print(f"Input features: {policy.config.input_features}")
    
    # Debug data storage
    debug_data = {
        "episodes": [],
        "summary": {
            "n_episodes": n_episodes,
            "batch_size": batch_size,
            "device": device,
            "checkpoint": checkpoint_path,
        }
    }
    
    # Run evaluation
    print(f"\nRunning {n_episodes} episodes...")
    policy.reset()
    
    observation, info = env.reset(seed=42)
    
    episode_data = {
        "rewards": [[] for _ in range(batch_size)],
        "successes": [[] for _ in range(batch_size)],
        "grasp_successes": [[] for _ in range(batch_size)],
        "actions": [[] for _ in range(batch_size)],
        "states": [[] for _ in range(batch_size)],
        "tasks": [],
    }
    
    start_time = time.time()
    step_count = 0
    max_steps = 400  # Max episode length
    
    with torch.no_grad():
        for step in range(max_steps):
            # Preprocess observation
            batch = preprocessor(observation)
            
            # Save task instruction
            if step == 0:
                tasks = batch.get("task", [""] * batch_size)
                episode_data["tasks"] = tasks if isinstance(tasks, list) else [tasks]
                print(f"\nTask instructions: {episode_data['tasks']}")
            
            # Select action
            action = policy.select_action(batch)
            
            # Save action and state
            action_np = action.cpu().numpy()
            state_np = batch["observation.state"].cpu().numpy() if "observation.state" in batch else None
            
            for i in range(batch_size):
                episode_data["actions"][i].append(action_np[i].tolist())
                if state_np is not None:
                    episode_data["states"][i].append(state_np[i].tolist())
            
            # Step environment
            observation, reward, terminated, truncated, step_info = env.step(action_np)
            
            # Save reward and success
            for i in range(batch_size):
                episode_data["rewards"][i].append(float(reward[i]))
                episode_data["successes"][i].append(bool(terminated[i]))
                episode_data["grasp_successes"][i].append(bool(step_info[i].get("grasp_success", False)))
            
            step_count += 1
            
            # Check if all episodes are done
            if all(terminated) or all(truncated):
                print(f"\nAll episodes completed at step {step}")
                break
            
            # Print progress every 50 steps
            if step % 50 == 0:
                elapsed = time.time() - start_time
                success_count = sum(1 for s in episode_data["successes"] if any(s))
                print(f"Step {step}/{max_steps} | Time: {elapsed:.1f}s | Successes: {success_count}/{batch_size}")
    
    # Calculate statistics
    print("\n" + "="*60)
    print("Episode Statistics:")
    print("="*60)
    
    for i in range(batch_size):
        total_reward = sum(episode_data["rewards"][i])
        max_reward = max(episode_data["rewards"][i]) if episode_data["rewards"][i] else 0
        success = any(episode_data["successes"][i])
        grasp_success = any(episode_data["grasp_successes"][i])
        
        # Action statistics
        actions = np.array(episode_data["actions"][i])
        action_mean = actions.mean(axis=0)
        action_std = actions.std(axis=0)
        action_min = actions.min(axis=0)
        action_max = actions.max(axis=0)
        
        print(f"\nEpisode {i}:")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Max reward: {max_reward:.2f}")
        print(f"  Success: {success}")
        print(f"  Grasp success: {grasp_success}")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Action mean: {action_mean}")
        print(f"  Action std: {action_std}")
        print(f"  Action min: {action_min}")
        print(f"  Action max: {action_max}")
        
        debug_data["episodes"].append({
            "episode_idx": i,
            "total_reward": total_reward,
            "max_reward": max_reward,
            "success": success,
            "grasp_success": grasp_success,
            "n_steps": len(episode_data["actions"][i]),
            "action_stats": {
                "mean": action_mean.tolist(),
                "std": action_std.tolist(),
                "min": action_min.tolist(),
                "max": action_max.tolist(),
            },
        })
    
    # Save debug data
    output_file = output_dir / "debug_eval_data.json"
    with open(output_file, "w") as f:
        json.dump(debug_data, f, indent=2)
    
    print(f"\n" + "="*60)
    print(f"Debug data saved to: {output_file}")
    print("="*60)
    
    # Save action sequences for visualization
    for i in range(batch_size):
        actions = np.array(episode_data["actions"][i])
        np.save(output_dir / f"episode_{i}_actions.npy", actions)
    
    print(f"Action sequences saved to: {output_dir}/episode_*.npy")
    
    # Close environment
    env.close()
    
    print("\nDone!")


if __name__ == "__main__":
    main()