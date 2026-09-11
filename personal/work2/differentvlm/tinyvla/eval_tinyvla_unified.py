#!/usr/bin/env python
"""
TinyVLA Unified Evaluation Script.

Evaluates TinyVLA with different action execution strategies and records
action smoothness metrics for fair comparison.

Strategies:
1. n_action_steps=16 (baseline)
2. n_action_steps=8
3. n_action_steps=4
4. n_action_steps=1 + temporal aggregation

Usage:
    python eval_tinyvla_unified.py \
        --policy_path=PATH \
        --env_type=metaworld \
        --env_task=disassemble-v3 \
        --env_camera_name=corner,gripperPOV \
        --env_use_self_mw=true \
        --eval_batch_size=8 \
        --eval_n_episodes=50 \
        --policy_device=cuda \
        --policy_use_amp=false \
        --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
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
    """
    
    def __init__(self, chunk_size: int = 16, decay: float = 0.01, num_envs: int = 1):
        self.chunk_size = chunk_size
        self.decay = decay
        self.num_envs = num_envs
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


def eval_with_n_action_steps(
    env,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    n_action_steps: int,
    chunk_size: int = 16,
    device: str = "cuda",
    start_seed: int | None = None,
    use_temporal_agg: bool = False,
    aggregation_decay: float = 0.01,
) -> dict:
    """
    Run evaluation with configurable n_action_steps.
    """
    num_envs = env.num_envs
    n_batches = n_episodes // num_envs + int((n_episodes % num_envs) != 0)
    
    aggregator = None
    if use_temporal_agg:
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
    
    strategy_name = f"n_action_steps={n_action_steps}"
    if use_temporal_agg:
        strategy_name += " + temporal_agg"
    
    for batch_idx in trange(n_batches, desc=f"Evaluating ({strategy_name})"):
        policy.reset()
        if aggregator:
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
        action_queue = [None] * num_envs
        queue_idx = [0] * num_envs
        
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
            
            # Determine if we need to predict a new chunk
            need_predict = [False] * num_envs
            for env_idx in range(num_envs):
                if not done[env_idx]:
                    if action_queue[env_idx] is None or queue_idx[env_idx] >= n_action_steps:
                        need_predict[env_idx] = True
            
            if any(need_predict):
                with torch.inference_mode():
                    action_chunk = policy.predict_action_chunk(observation)
                
                action_chunk_np = action_chunk.to("cpu").to(dtype=torch.float32).numpy()
                
                for env_idx in range(num_envs):
                    if need_predict[env_idx] and not done[env_idx]:
                        if use_temporal_agg and aggregator:
                            aggregator.add_chunk(env_idx, action_chunk_np[env_idx])
                            
                            agg_action = aggregator.get_aggregated_action(env_idx)
                            if agg_action is None:
                                agg_action = action_chunk_np[env_idx, 0]
                            
                            action_queue[env_idx] = np.stack([agg_action] * chunk_size)
                        else:
                            action_queue[env_idx] = action_chunk_np[env_idx]
                        
                        queue_idx[env_idx] = 0
            
            # Get current action from queue
            current_actions = []
            for env_idx in range(num_envs):
                if done[env_idx]:
                    current_actions.append(np.zeros(action_queue[0].shape[-1]))
                else:
                    current_actions.append(action_queue[env_idx][queue_idx[env_idx]])
            
            current_actions = np.stack(current_actions)
            
            # Track action smoothness metrics
            for env_idx in range(num_envs):
                if not done[env_idx]:
                    if prev_actions[env_idx] is not None:
                        action_diff = np.linalg.norm(
                            current_actions[env_idx] - prev_actions[env_idx]
                        )
                        episode_action_diffs[env_idx].append(action_diff)
                    
                    gripper_val = current_actions[env_idx, -1]
                    if prev_gripper[env_idx] is not None:
                        if (prev_gripper[env_idx] > 0 and gripper_val < 0) or \
                           (prev_gripper[env_idx] < 0 and gripper_val > 0):
                            episode_gripper_flips[env_idx] += 1
                    
                    prev_actions[env_idx] = current_actions[env_idx].copy()
                    prev_gripper[env_idx] = gripper_val
            
            # Postprocess actions
            action_tensor = torch.from_numpy(current_actions).to(device)
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
                    
                    if aggregator:
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
                    queue_idx[env_idx] += 1
                    if aggregator:
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
    logger.info(f"Evaluation Results: {strategy_name}")
    logger.info("=" * 80)
    logger.info(f"Success rate: {success_rate:.2%}")
    logger.info(f"Avg sum reward: {avg_sum_reward:.4f}")
    logger.info(f"Avg max reward: {avg_max_reward:.4f}")
    logger.info(f"Mean action diff (||a_t - a_{{t-1}}||): {mean_action_diff:.4f}")
    logger.info(f"Mean gripper flips: {mean_gripper_flips:.2f}")
    logger.info(f"Total episodes: {len(all_successes)}")
    logger.info("=" * 80)
    
    return {
        "strategy": strategy_name,
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
    parser = argparse.ArgumentParser(description="TinyVLA Unified Evaluation")
    
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
    parser.add_argument("--strategies", type=str, default="16,8,4,1+agg",
                        help="Comma-separated strategies: 16,8,4,1+agg")
    
    args = parser.parse_args()
    
    init_logging()
    register_third_party_plugins()
    
    rename_map = json.loads(args.rename_map) if args.rename_map else {}
    
    logger.info("=" * 80)
    logger.info("TinyVLA Unified Evaluation")
    logger.info("=" * 80)
    logger.info(f"Policy path: {args.policy_path}")
    logger.info(f"Device: {args.policy_device}")
    logger.info(f"Environment: {args.env_type} - {args.env_task}")
    logger.info(f"Cameras: {args.env_camera_name}")
    logger.info(f"Batch size: {args.eval_batch_size}")
    logger.info(f"Episodes: {args.eval_n_episodes}")
    logger.info(f"Strategies: {args.strategies}")
    logger.info("=" * 80)
    
    device = get_safe_torch_device(args.policy_device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)
    
    if args.policy_device.startswith("cuda:"):
        gpu_id = args.policy_device.split(":")[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    elif args.policy_device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    from lerobot.envs.configs import MetaworldEnv
    env_cfg = MetaworldEnv(
        task=args.env_task,
        camera_name=args.env_camera_name,
        use_self_mw=args.env_use_self_mw,
    )
    
    envs_dict = make_env(env_cfg, n_envs=args.eval_batch_size)
    env = list(envs_dict.values())[0][0]
    
    from lerobot.configs.policies import PreTrainedConfig
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    policy_cfg.device = args.policy_device
    policy_cfg.use_amp = args.policy_use_amp
    
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=rename_map)
    policy.eval()
    
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
    
    strategies = [s.strip() for s in args.strategies.split(",")]
    all_results = []
    
    for strategy in strategies:
        if strategy == "1+agg":
            n_action_steps = 1
            use_temporal_agg = True
        else:
            n_action_steps = int(strategy)
            use_temporal_agg = False
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing strategy: n_action_steps={n_action_steps}" + 
                   (" + temporal aggregation" if use_temporal_agg else ""))
        logger.info(f"{'='*80}\n")
        
        with torch.no_grad():
            result = eval_with_n_action_steps(
                env=env,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=args.eval_n_episodes,
                n_action_steps=n_action_steps,
                chunk_size=args.chunk_size,
                device=str(device),
                start_seed=args.seed,
                use_temporal_agg=use_temporal_agg,
                aggregation_decay=args.aggregation_decay,
            )
        
        all_results.append(result)
    
    close_envs(envs_dict)
    
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Strategy':<30} {'Success%':<10} {'Action Diff':<15} {'Gripper Flips':<15}")
    logger.info("-" * 80)
    for result in all_results:
        logger.info(f"{result['strategy']:<30} {result['pc_success']:<10.2f} {result['mean_action_diff']:<15.4f} {result['mean_gripper_flips']:<15.2f}")
    logger.info("=" * 80)
    
    logger.info("End of eval")


if __name__ == "__main__":
    main()