#!/usr/bin/env python
"""
TinyVLA Temporal Aggregation Evaluation Script.

This script evaluates TinyVLA with temporal aggregation to reduce action jitter
compared to single-step (n_action_steps=1) execution.

Temporal aggregation works by:
1. At each step, calling predict_action_chunk() to get a full 16-step action chunk
2. Maintaining a history of the last 16 overlapping chunks per environment
3. For the current timestep, collecting all predictions that cover it
4. Computing a weighted average using exp(-0.01 * age) weights
5. Applying postprocessor and env bounds clipping after aggregation

Usage:
    bash personal/work2/differentvlm/tinyvla/run_eval_temporal_aggregation.sh
"""

import os
import sys
import json
import logging
import time
import argparse
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import trange

from lerobot.envs.utils import close_envs
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy, init_logging, register_third_party_plugins
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed
from lerobot.utils.constants import ACTION

logger = logging.getLogger(__name__)


class TemporalAggregator:
    """
    Temporal aggregation for action chunks.
    
    Maintains a history of recent action chunks and aggregates predictions
    for the current timestep using exponential weighting.
    """
    
    def __init__(self, chunk_size: int = 16, decay: float = 0.01, num_envs: int = 1):
        self.chunk_size = chunk_size
        self.decay = decay
        self.num_envs = num_envs
        
        # Per-environment history: deque of (chunk, timestamp)
        self.histories = [deque(maxlen=chunk_size) for _ in range(num_envs)]
        self.current_steps = [0] * num_envs
    
    def reset(self, env_idx: int | None = None):
        if env_idx is not None:
            self.histories[env_idx].clear()
            self.current_steps[env_idx] = 0
        else:
            for i in range(self.num_envs):
                self.histories[i].clear()
                self.current_steps[i] = 0
    
    def add_chunk(self, env_idx: int, chunk: np.ndarray):
        self.histories[env_idx].append({
            'chunk': chunk.copy(),
            'timestamp': self.current_steps[env_idx],
        })
    
    def get_aggregated_action(self, env_idx: int) -> np.ndarray | None:
        current_step = self.current_steps[env_idx]
        history = self.histories[env_idx]
        
        if len(history) == 0:
            return None
        
        weighted_sum = None
        weight_sum = 0.0
        
        for item in history:
            chunk = item['chunk']
            timestamp = item['timestamp']
            step_in_chunk = current_step - timestamp
            
            if 0 <= step_in_chunk < self.chunk_size:
                age = current_step - timestamp
                weight = np.exp(-self.decay * age)
                action = chunk[step_in_chunk]
                
                if weighted_sum is None:
                    weighted_sum = weight * action
                else:
                    weighted_sum += weight * action
                
                weight_sum += weight
        
        if weight_sum > 0:
            return weighted_sum / weight_sum
        return None
    
    def step(self, env_idx: int | None = None):
        if env_idx is not None:
            self.current_steps[env_idx] += 1
        else:
            for i in range(self.num_envs):
                self.current_steps[i] += 1


def eval_with_temporal_aggregation(
    env,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    chunk_size: int = 16,
    aggregation_decay: float = 0.01,
    device: str = "cuda",
    start_seed: int | None = None,
) -> dict:
    """
    Run evaluation with temporal aggregation.
    """
    num_envs = env.num_envs
    n_batches = n_episodes // num_envs + int((n_episodes % num_envs) != 0)
    
    aggregator = TemporalAggregator(
        chunk_size=chunk_size,
        decay=aggregation_decay,
        num_envs=num_envs,
    )
    
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_action_diffs = []
    all_gripper_flips = []
    
    for batch_idx in trange(n_batches, desc="Evaluating with temporal aggregation"):
        policy.reset()
        aggregator.reset()
        
        seeds = None
        if start_seed is not None:
            seeds = list(range(start_seed + batch_idx * num_envs, start_seed + (batch_idx + 1) * num_envs))
        
        observation, info = env.reset(seed=seeds)
        
        done = np.array([False] * num_envs)
        max_steps = env.call("_max_episode_steps")[0]
        
        episode_rewards = np.zeros(num_envs)
        episode_max_rewards = np.zeros(num_envs)
        prev_actions = [None] * num_envs
        prev_gripper = [None] * num_envs
        episode_action_diffs = [[] for _ in range(num_envs)]
        episode_gripper_flips = [0 for _ in range(num_envs)]
        
        step = 0
        
        while not np.all(done) and step < max_steps:
            observation = preprocess_observation(observation)
            
            try:
                observation["task"] = list(env.call("task_description"))
            except (AttributeError, NotImplementedError):
                try:
                    observation["task"] = list(env.call("task"))
                except (AttributeError, NotImplementedError):
                    observation["task"] = [""] * num_envs
            
            observation = env_preprocessor(observation)
            observation = preprocessor(observation)
            
            for k, v in observation.items():
                if isinstance(v, torch.Tensor):
                    observation[k] = v.to(device)
            
            # Predict action chunk using policy's predict_action_chunk method
            with torch.inference_mode():
                action_chunk = policy.predict_action_chunk(observation)
            
            # Convert to numpy: (batch, chunk_size, action_dim)
            action_chunk_np = action_chunk.to("cpu").to(dtype=torch.float32).numpy()
            
            # Add chunks to aggregator
            for env_idx in range(num_envs):
                if not done[env_idx]:
                    aggregator.add_chunk(env_idx, action_chunk_np[env_idx])
            
            # Get aggregated actions
            aggregated_actions = []
            for env_idx in range(num_envs):
                if done[env_idx]:
                    aggregated_actions.append(np.zeros(action_chunk_np.shape[-1]))
                else:
                    agg_action = aggregator.get_aggregated_action(env_idx)
                    if agg_action is None:
                        agg_action = action_chunk_np[env_idx, 0]
                    aggregated_actions.append(agg_action)
            
            aggregated_actions = np.stack(aggregated_actions)
            
            # Track action smoothness metrics
            for env_idx in range(num_envs):
                if not done[env_idx]:
                    if prev_actions[env_idx] is not None:
                        action_diff = np.linalg.norm(
                            aggregated_actions[env_idx] - prev_actions[env_idx]
                        )
                        episode_action_diffs[env_idx].append(action_diff)
                    
                    gripper_val = aggregated_actions[env_idx, -1]
                    if prev_gripper[env_idx] is not None:
                        if (prev_gripper[env_idx] > 0 and gripper_val < 0) or \
                           (prev_gripper[env_idx] < 0 and gripper_val > 0):
                            episode_gripper_flips[env_idx] += 1
                    
                    prev_actions[env_idx] = aggregated_actions[env_idx].copy()
                    prev_gripper[env_idx] = gripper_val
            
            # Postprocess aggregated actions
            action_tensor = torch.from_numpy(aggregated_actions).to(device)
            action_transition = {ACTION: action_tensor}
            action_transition = env_postprocessor(action_transition)
            action_tensor = action_transition[ACTION]
            action_tensor = postprocessor(action_tensor)
            
            action_numpy = action_tensor.to("cpu").to(dtype=torch.float32).numpy()
            
            env_low = env.action_space.low
            env_high = env.action_space.high
            action_numpy = np.clip(action_numpy, env_low, env_high)
            
            observation, reward, terminated, truncated, info = env.step(action_numpy)
            
            episode_rewards += reward
            episode_max_rewards = np.maximum(episode_max_rewards, reward)
            
            if "final_info" in info:
                final_info = info["final_info"]
                if isinstance(final_info, dict):
                    is_success = final_info.get("is_success", [False] * num_envs)
                    successes = is_success.tolist() if hasattr(is_success, "tolist") else [bool(is_success)] * num_envs
                else:
                    successes = []
                    for item in final_info:
                        if isinstance(item, dict) and "is_success" in item:
                            successes.append(bool(item["is_success"]))
                        else:
                            successes.append(False)
            else:
                successes = [False] * num_envs
            
            newly_done = terminated | truncated
            
            for env_idx in range(num_envs):
                if newly_done[env_idx] and not done[env_idx]:
                    sum_rewards.append(float(episode_rewards[env_idx]))
                    max_rewards.append(float(episode_max_rewards[env_idx]))
                    all_successes.append(successes[env_idx])
                    all_action_diffs.extend(episode_action_diffs[env_idx])
                    all_gripper_flips.append(episode_gripper_flips[env_idx])
                    
                    aggregator.reset(env_idx)
                    prev_actions[env_idx] = None
                    prev_gripper[env_idx] = None
                    episode_rewards[env_idx] = 0
                    episode_max_rewards[env_idx] = 0
                    episode_action_diffs[env_idx] = []
                    episode_gripper_flips[env_idx] = 0
            
            done = newly_done | done
            if step + 1 == max_steps:
                done = np.ones_like(done, dtype=bool)
            
            for env_idx in range(num_envs):
                if not done[env_idx]:
                    aggregator.step(env_idx)
            
            step += 1
        
        for env_idx in range(num_envs):
            if not done[env_idx]:
                sum_rewards.append(float(episode_rewards[env_idx]))
                max_rewards.append(float(episode_max_rewards[env_idx]))
                all_successes.append(False)
                all_action_diffs.extend(episode_action_diffs[env_idx])
                all_gripper_flips.append(episode_gripper_flips[env_idx])
    
    success_rate = np.mean(all_successes) if all_successes else 0.0
    avg_sum_reward = np.mean(sum_rewards) if sum_rewards else 0.0
    avg_max_reward = np.mean(max_rewards) if max_rewards else 0.0
    mean_action_diff = np.mean(all_action_diffs) if all_action_diffs else 0.0
    mean_gripper_flips = np.mean(all_gripper_flips) if all_gripper_flips else 0.0
    
    logger.info("=" * 80)
    logger.info("Temporal Aggregation Evaluation Results")
    logger.info("=" * 80)
    logger.info(f"Success rate: {success_rate:.2%}")
    logger.info(f"Avg sum reward: {avg_sum_reward:.4f}")
    logger.info(f"Avg max reward: {avg_max_reward:.4f}")
    logger.info(f"Mean action diff (||a_t - a_{{t-1}}||): {mean_action_diff:.4f}")
    logger.info(f"Mean gripper flips: {mean_gripper_flips:.2f}")
    logger.info(f"Total episodes: {len(all_successes)}")
    logger.info("=" * 80)
    
    return {
        "sum_rewards": sum_rewards,
        "max_rewards": max_rewards,
        "successes": all_successes,
        "action_diffs": all_action_diffs,
        "gripper_flips": all_gripper_flips,
        "avg_sum_reward": avg_sum_reward,
        "avg_max_reward": avg_max_reward,
        "pc_success": success_rate * 100,
        "mean_action_diff": mean_action_diff,
        "mean_gripper_flips": mean_gripper_flips,
    }


def preprocess_observation(observation):
    """Convert numpy observation to torch tensors."""
    processed = {}
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            if value.dtype == np.uint8:
                processed[key] = torch.from_numpy(value).float() / 255.0
            else:
                processed[key] = torch.from_numpy(value).float()
        elif isinstance(value, torch.Tensor):
            processed[key] = value
        elif isinstance(value, list):
            processed[key] = value
        else:
            processed[key] = value
    return processed


def main():
    parser = argparse.ArgumentParser(description="TinyVLA Temporal Aggregation Evaluation")
    
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--policy_device", type=str, default="cuda")
    parser.add_argument("--policy_use_amp", type=bool, default=False)
    parser.add_argument("--env_type", type=str, required=True)
    parser.add_argument("--env_task", type=str, required=True)
    parser.add_argument("--env_camera_name", type=str, default="corner,gripperPOV")
    parser.add_argument("--env_use_self_mw", type=bool, default=True)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_n_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--aggregation_decay", type=float, default=0.01)
    parser.add_argument("--rename_map", type=str, default=None)
    
    args = parser.parse_args()
    
    init_logging()
    register_third_party_plugins()
    
    rename_map = json.loads(args.rename_map) if args.rename_map else {}
    
    logger.info("=" * 80)
    logger.info("TinyVLA Temporal Aggregation Evaluation")
    logger.info("=" * 80)
    logger.info(f"Policy path: {args.policy_path}")
    logger.info(f"Device: {args.policy_device}")
    logger.info(f"Environment: {args.env_type} - {args.env_task}")
    logger.info(f"Cameras: {args.env_camera_name}")
    logger.info(f"Batch size: {args.eval_batch_size}")
    logger.info(f"Episodes: {args.eval_n_episodes}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Aggregation decay: {args.aggregation_decay}")
    logger.info("=" * 80)
    
    device = get_safe_torch_device(args.policy_device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)
    
    # Set CUDA_VISIBLE_DEVICES
    if args.policy_device.startswith("cuda:"):
        gpu_id = args.policy_device.split(":")[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    elif args.policy_device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Create environment config
    from lerobot.envs.configs import MetaworldEnv
    env_cfg = MetaworldEnv(
        task=args.env_task,
        camera_name=args.env_camera_name,
        use_self_mw=args.env_use_self_mw,
    )
    
    # Make environment
    envs_dict = make_env(env_cfg, n_envs=args.eval_batch_size)
    # envs_dict is {suite_name: {task_id: vec_env}}
    env = list(envs_dict.values())[0][0]
    
    # Make policy
    from lerobot.configs.policies import PreTrainedConfig
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    policy_cfg.device = args.policy_device
    policy_cfg.use_amp = args.policy_use_amp
    
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=rename_map)
    policy.eval()
    
    # Make processors
    preprocessor_overrides = {
        "device_processor": {"device": str(policy_cfg.device)},
        "rename_observations_processor": {"rename_map": rename_map},
    }
    
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    
    # Run evaluation
    with torch.no_grad():
        results = eval_with_temporal_aggregation(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=args.eval_n_episodes,
            chunk_size=args.chunk_size,
            aggregation_decay=args.aggregation_decay,
            device=str(device),
            start_seed=args.seed,
        )
    
    close_envs(envs_dict)
    logger.info("End of eval")


if __name__ == "__main__":
    main()