#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import metaworld
import metaworld.policies as policies
import numpy as np
from gymnasium import spaces

from lerobot.lerobot_types import RobotObservation

from .utils import _LazyAsyncVectorEnv

# ---- Load configuration data from the external JSON file ----
CONFIG_PATH = Path(__file__).parent / "metaworld_config.json"
try:
    with open(CONFIG_PATH) as f:
        data = json.load(f)
except FileNotFoundError as err:
    raise FileNotFoundError(
        "Could not find 'metaworld_config.json'. "
        "Please ensure the configuration file is in the same directory as the script."
    ) from err
except json.JSONDecodeError as err:
    raise ValueError(
        "Failed to decode 'metaworld_config.json'. Please ensure it is a valid JSON file."
    ) from err

# ---- Process the loaded data ----

# extract and type-check top-level dicts
task_descriptions_obj = data.get("TASK_DESCRIPTIONS")
if not isinstance(task_descriptions_obj, dict):
    raise TypeError("Expected TASK_DESCRIPTIONS to be a dict[str, str]")
TASK_DESCRIPTIONS: dict[str, str] = task_descriptions_obj

task_name_to_id_obj = data.get("TASK_NAME_TO_ID")
if not isinstance(task_name_to_id_obj, dict):
    raise TypeError("Expected TASK_NAME_TO_ID to be a dict[str, int]")
TASK_NAME_TO_ID: dict[str, int] = task_name_to_id_obj

# difficulty -> tasks mapping
difficulty_to_tasks = data.get("DIFFICULTY_TO_TASKS")
if not isinstance(difficulty_to_tasks, dict):
    raise TypeError("Expected 'DIFFICULTY_TO_TASKS' to be a dict[str, list[str]]")
DIFFICULTY_TO_TASKS: dict[str, list[str]] = difficulty_to_tasks

# convert policy strings -> actual policy classes
task_policy_mapping = data.get("TASK_POLICY_MAPPING")
if not isinstance(task_policy_mapping, dict):
    raise TypeError("Expected 'TASK_POLICY_MAPPING' to be a dict[str, str]")
TASK_POLICY_MAPPING: dict[str, Any] = {
    task_name: getattr(policies, policy_class_name)
    for task_name, policy_class_name in task_policy_mapping.items()
}
ACTION_DIM = 4
OBS_DIM = 4


class MetaworldEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        task,
        camera_name="corner2",
        obs_type="pixels",
        render_mode="rgb_array",
        observation_width=480,
        observation_height=480,
        visualization_width=640,
        visualization_height=480,
        fixed_states: list[np.ndarray] | None = None,
    ):
        super().__init__()
        self.task = task.replace("metaworld-", "")
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height

        # Support multiple cameras (comma-separated, e.g., "corner2,behindGripper")
        self.camera_names = [c.strip() for c in camera_name.split(",")]
        self.camera_name = camera_name  # Keep original for backward compatibility

        self._fixed_states = fixed_states
        self._fixed_state_cursor = 0

        self._env_name = self.task  # already stripped of "metaworld-" prefix above
        self._env = None  # deferred — created on first reset() inside the worker subprocess
        self._env_wrist = None  # second camera for dual-camera setup
        self._max_episode_steps = 500  # MT1 environments always have max_path_length=500
        self.task_description = TASK_DESCRIPTIONS[self.task]

        self.expert_policy = TASK_POLICY_MAPPING[self.task]()

        if self.obs_type == "state":
            raise NotImplementedError()
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    )
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    ),
                    "agent_pos": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(OBS_DIM,),
                        dtype=np.float64,
                    ),
                }
            )

        self.action_space = spaces.Box(low=-1, high=1, shape=(ACTION_DIM,), dtype=np.float32)

    def _ensure_env(self) -> None:
        """Create the underlying MetaWorld env on first use.

        Called inside the worker subprocess after fork(), so each worker gets
        its own clean rendering context rather than inheriting a stale one from
        the parent process (which causes crashes with AsyncVectorEnv).
        """
        if self._env is not None:
            return
        mt1 = metaworld.MT1(self._env_name, seed=42)
        env = mt1.train_classes[self._env_name](render_mode="rgb_array", camera_name=self.camera_names[0])
        env.set_task(mt1.train_tasks[0])
        if self.camera_names[0] == "corner2":
            env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        env.reset()
        env._freeze_rand_vec = False  # otherwise no randomization
        env.seeded_rand_vec = True  # use seeded RNG so reset(seed=X) controls object positions
        self._env = env

        # Create second camera environment if dual-camera mode
        if len(self.camera_names) > 1:
            env_wrist = mt1.train_classes[self._env_name](render_mode="rgb_array", camera_name=self.camera_names[1])
            env_wrist.set_task(mt1.train_tasks[0])
            env_wrist.reset()
            env_wrist._freeze_rand_vec = False
            env_wrist.seeded_rand_vec = True
            self._env_wrist = env_wrist

    def render(self) -> np.ndarray:
        """
        Render the current environment frame.

        Returns:
            np.ndarray: The rendered RGB image from the environment.
        """
        self._ensure_env()
        image = self._env.render()
        if self.camera_names[0] == "corner2":
            # Images from this camera are flipped — correct them
            image = np.flip(image, (0, 1))
        return image

    def _sync_env_state(self, src_env, dst_env):
        """Sync simulation state from src to dst environment."""
        dst_env.data.qpos[:] = src_env.data.qpos[:]
        dst_env.data.qvel[:] = src_env.data.qvel[:]
        dst_env.data.ctrl[:] = src_env.data.ctrl[:]
        mujoco.mj_forward(dst_env.model, dst_env.data)

    def _render_camera(self, env, camera_name) -> np.ndarray:
        """Render a specific camera view."""
        image = env.render()
        if camera_name == "corner2":
            image = np.flip(image, (0, 1))
        return image

    def _format_raw_obs(self, raw_obs: np.ndarray) -> RobotObservation:
        image = None
        if self._env is not None:
            image = self._render_camera(self._env, self.camera_names[0])

        # Sync wrist camera environment state and render
        image_wrist = None
        if self._env_wrist is not None:
            self._sync_env_state(self._env, self._env_wrist)
            image_wrist = self._render_camera(self._env_wrist, self.camera_names[1])

        agent_pos = raw_obs[:4]
        if self.obs_type == "state":
            raise NotImplementedError(
                "'state' obs_type not implemented for MetaWorld. Use pixel modes instead."
            )

        elif self.obs_type in ("pixels", "pixels_agent_pos"):
            assert image is not None, (
                "Expected `image` to be rendered before constructing pixel-based observations. "
                "This likely means `env.render()` returned None or the environment was not provided."
            )

            if self.obs_type == "pixels":
                obs = {"pixels/top": image.copy()}

            else:  # pixels_agent_pos
                obs = {
                    "pixels/top": image.copy(),
                    "agent_pos": agent_pos,
                }

            # Add wrist camera image if dual-camera mode
            if image_wrist is not None:
                obs["pixels/wrist"] = image_wrist.copy()
        else:
            raise ValueError(f"Unknown obs_type: {self.obs_type}")
        return obs

    def reset(
        self,
        seed: int | None = None,
        **kwargs,
    ) -> tuple[RobotObservation, dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Args:
            seed (Optional[int]): Random seed for environment initialization.

        Returns:
            observation (RobotObservation): The initial formatted observation.
            info (Dict[str, Any]): Additional info about the reset state.
        """
        self._ensure_env()
        super().reset(seed=seed)

        if self._fixed_states is not None:
            rand_vec = self._fixed_states[self._fixed_state_cursor % len(self._fixed_states)]
            self._fixed_state_cursor += 1
            self._env._freeze_rand_vec = False
            self._env._get_state_rand_vec = lambda: rand_vec.copy()

        if seed is not None:
            self._env.seed(seed)
        raw_obs, info = self._env.reset(seed=seed)

        # Sync wrist camera if dual-camera mode
        if self._env_wrist is not None:
            if seed is not None:
                self._env_wrist.seed(seed)
            self._env_wrist.reset(seed=seed)
            self._sync_env_state(self._env, self._env_wrist)

        observation = self._format_raw_obs(raw_obs)

        info = {
            "is_success": False,
            "obj_init_pos": self._env.obj_init_pos.copy() if self._env.obj_init_pos is not None else None,
            "goal_pos": self._env.goal.copy(),
        }
        return observation, info

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        """
        Perform one environment step.

        Args:
            action (np.ndarray): The action to execute, must be 1-D with shape (action_dim,).

        Returns:
            observation (RobotObservation): The formatted observation after the step.
            reward (float): The scalar reward for this step.
            terminated (bool): Whether the episode terminated successfully.
            truncated (bool): Whether the episode was truncated due to a time limit.
            info (Dict[str, Any]): Additional environment info.
        """
        self._ensure_env()
        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )
        raw_obs, reward, done, truncated, info = self._env.step(action)

        # Determine whether the task was successful
        is_success = bool(info.get("success", 0))
        terminated = done or is_success
        info.update(
            {
                "task": self.task,
                "done": done,
                "is_success": is_success,
            }
        )

        # Format the raw observation into the expected structure
        observation = self._format_raw_obs(raw_obs)
        if terminated:
            info["final_info"] = {
                "task": self.task,
                "done": bool(done),
                "is_success": bool(is_success),
            }
            self.reset()

        return observation, reward, terminated, truncated, info

    def close(self):
        if self._env is not None:
            self._env.close()
        if self._env_wrist is not None:
            self._env_wrist.close()


# ---- Main API ----------------------------------------------------------------


def create_metaworld_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> dict[str, dict[int, Any]]:
    """
    Create vectorized Meta-World environments with a consistent return shape.

    Returns:
        dict[task_group][task_id] -> vec_env (env_cls([...]) with exactly n_envs factories)
    Notes:
        - n_envs is the number of rollouts *per task* (episode_index = 0..n_envs-1).
        - `task` can be a single difficulty group (e.g., "easy", "medium", "hard") or a comma-separated list.
        - If a task name is not in DIFFICULTY_TO_TASKS, we treat it as a single custom task.
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    task_groups = [t.strip() for t in task.split(",") if t.strip()]
    if not task_groups:
        raise ValueError("`task` must contain at least one Meta-World task or difficulty group.")

    print(f"Creating Meta-World envs | task_groups={task_groups} | n_envs(per task)={n_envs}")

    is_async = env_cls is gym.vector.AsyncVectorEnv
    cached_obs_space = None
    cached_act_space = None
    cached_metadata = None
    out: dict[str, dict[int, Any]] = defaultdict(dict)

    for group in task_groups:
        # if not in difficulty presets, treat it as a single custom task
        tasks = DIFFICULTY_TO_TASKS.get(group, [group])

        for tid, task_name in enumerate(tasks):
            print(f"Building vec env | group={group} | task_id={tid} | task={task_name}")

            # build n_envs factories
            fns = [(lambda tn=task_name: MetaworldEnv(task=tn, **gym_kwargs)) for _ in range(n_envs)]

            if is_async:
                lazy = _LazyAsyncVectorEnv(fns, cached_obs_space, cached_act_space, cached_metadata)
                if cached_obs_space is None:
                    cached_obs_space = lazy.observation_space
                    cached_act_space = lazy.action_space
                    cached_metadata = lazy.metadata
                out[group][tid] = lazy
            else:
                out[group][tid] = env_cls(fns)

    # return a plain dict for consistency
    return {group: dict(task_map) for group, task_map in out.items()}