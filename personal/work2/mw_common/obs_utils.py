"""Meta-World 39维 observation.environment_state 的解析工具(修正版,见 SPEC.md 3.1节)。"""

import numpy as np


ENV_STATE_LAYOUT = {
    "hand_pos": slice(0, 3),
    "gripper": slice(3, 4),
    "obj1_pos": slice(4, 7),
    "obj1_quat": slice(7, 11),
    "obj2_pos": slice(11, 14),
    "obj2_quat": slice(14, 18),
    "prev_frame_stack": slice(18, 36),
    "goal_pos": slice(36, 39),
}


def obj_pos(env_state):
    """从39维environment_state中提取物体位置。"""
    return env_state[..., ENV_STATE_LAYOUT["obj1_pos"]]


def goal_pos(env_state):
    """从39维environment_state中提取目标位置。"""
    return env_state[..., ENV_STATE_LAYOUT["goal_pos"]]


def find_task_index(dataset_meta, task_description_substring: str) -> int:
    """按任务描述文本匹配 task_index,不要硬编码数字(不同数据集的编号可能不同)。"""
    tasks = getattr(dataset_meta, "tasks", None)
    if tasks is None:
        raise AttributeError("dataset_meta 没有 'tasks' 属性,请先 print(dataset_meta) 确认结构")
    for idx, desc in tasks.items():
        if task_description_substring in str(desc):
            return idx
    raise ValueError(f"未找到匹配 '{task_description_substring}' 的任务")