"""
根据已有 LeRobot 数据集重新生成 episode_initial_states.json

使用 LeRobotDataset 类读取数据集，根据每个 episode 第一帧的
observation.environment_state 恢复环境初始状态。
支持多任务混合数据集（如 coffee-push-v3 + pick-place-v3）。

不修改数据集中的 parquet、video、meta 等任何原始数据，只生成新的 JSON 文件。
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_ENV_STATE_DESCRIPTION = {
    "0:3": "末端执行器(手)位置 xyz",
    "3:4": "夹爪开合度(归一化)",
    "4:7": "物体1位置 xyz (= obj_pose)",
    "7:11": "物体1四元数朝向(4维)",
    "11:14": "物体2位置(单物体任务中恒为0)",
    "14:18": "物体2四元数(恒为0)",
    "18:36": "上一帧的[0:18]原样重复(frame-stack)",
    "36:39": "目标位置 xyz (= goal_pose)",
}


def extract_env_state_value(frame: dict, key: str) -> np.ndarray | None:
    val = frame.get(key)
    if val is None:
        return None
    if hasattr(val, "numpy"):
        val = val.numpy()
    if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
        return np.asarray(val, dtype=np.float32)
    return None


def recover_episode_state(
    ep_idx: int,
    frame: dict,
    task_name: str,
    task_index: int,
    ep_length: int,
) -> dict:
    env_state = extract_env_state_value(frame, "observation.environment_state")
    success = frame.get("next.success")

    entry = {
        "episode_index": ep_idx,
        "task": task_name,
        "task_index": int(task_index),
        "num_frames": int(ep_length),
    }
    missing_fields = []

    if env_state is not None and len(env_state) >= 39:
        entry["obj_init_pos"] = env_state[4:7].tolist()
        entry["goal_pose"] = env_state[36:39].tolist()
    else:
        missing_fields.extend(["obj_init_pos", "goal_pose"])
        logger.warning(f"  Episode {ep_idx}: environment_state 无效或缺失, 无法恢复 obj_init_pos/goal_pose")

    if success is not None:
        if hasattr(success, "item"):
            success = bool(success.item())
        elif hasattr(success, "__iter__") and not isinstance(success, (str, bytes)):
            success = bool(np.asarray(success)[0])
        entry["success"] = bool(success)
    else:
        entry["success"] = False

    rand_vec = frame.get("rand_vec")
    if rand_vec is not None:
        if hasattr(rand_vec, "tolist"):
            rand_vec = rand_vec.tolist()
        entry["rand_vec"] = rand_vec
    else:
        entry["rand_vec_missing"] = True

    if missing_fields:
        entry["missing_fields"] = missing_fields

    return entry


def detect_env_state_structure(dataset: LeRobotDataset) -> dict:
    """从数据集中动态检测 environment_state 的结构。"""
    try:
        first_frame = dataset.get_raw_item(0)
        env_state = extract_env_state_value(first_frame, "observation.environment_state")

        if env_state is None or len(env_state) == 0:
            return DEFAULT_ENV_STATE_DESCRIPTION

        dim = len(env_state)
        structure = {"dimension": dim}

        if dim >= 39:
            structure.update(DEFAULT_ENV_STATE_DESCRIPTION)
        else:
            structure["note"] = f"environment_state 维度为 {dim}，与标准 39 维不同，结构描述可能不完整"

        return structure
    except Exception as e:
        logger.warning(f"检测 environment_state 结构时出错: {e}，使用默认描述")
        return DEFAULT_ENV_STATE_DESCRIPTION


def rebuild_episodeInitialStates(
    dataset_path: str,
    output_path: str | None = None,
    dry_run: bool = False,
) -> dict:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        logger.error(f"数据集目录不存在: {dataset_path}")
        sys.exit(1)

    logger.info(f"读取数据集路径: {dataset_path}")

    dataset = LeRobotDataset(
        repo_id=dataset_path.name,
        root=str(dataset_path),
    )

    total_episodes_info = dataset.meta.total_episodes
    total_frames_info = dataset.meta.total_frames
    total_tasks_info = dataset.meta.total_tasks
    logger.info(f"meta/info.json: total_episodes={total_episodes_info}, total_frames={total_frames_info}, total_tasks={total_tasks_info}")

    tasks_map = {}
    if dataset.meta.tasks is not None:
        for task_name, row in dataset.meta.tasks.iterrows():
            idx = int(row["task_index"])
            tasks_map[idx] = task_name
    logger.info(f"加载 {len(tasks_map)} 个任务映射: {tasks_map}")

    env_state_structure = detect_env_state_structure(dataset)
    logger.info(f"检测到 environment_state 结构: {env_state_structure}")

    episode_meta_list = dataset.meta.episodes
    actual_episodes = len(episode_meta_list)
    logger.info(f"从 dataset.meta.episodes 发现实际 episode 数量: {actual_episodes}")

    if actual_episodes != total_episodes_info:
        logger.warning(
            f"Episode 数量不一致: info.json 报告 {total_episodes_info}, "
            f"但 episodes 包含 {actual_episodes} 个 episode。"
            f"将以实际 episodes 为准 ({actual_episodes})。"
        )

    task_counter = {}
    recovered_count = 0
    missing_stats = {"obj_init_pos": 0, "goal_pose": 0, "rand_vec": 0}
    episodes_list = []

    for ep_meta in episode_meta_list:
        ep_idx = int(ep_meta["episode_index"])
        task_idx = int(ep_meta.get("task_index", -1))
        task_name = tasks_map.get(task_idx, f"unknown_task_{task_idx}")
        ep_length = int(ep_meta.get("length", 0))
        task_counter[task_name] = task_counter.get(task_name, 0) + 1

        try:
            from_idx = int(ep_meta.get("dataset_from_index", -1))
            if from_idx < 0:
                raise ValueError(f"episode {ep_idx} 的 dataset_from_index 无效: {from_idx}")

            first_frame = dataset.get_raw_item(from_idx)
            entry = recover_episode_state(ep_idx, first_frame, task_name, task_idx, ep_length)
            episodes_list.append(entry)

            if "obj_init_pos" in entry and "goal_pose" in entry:
                recovered_count += 1
            if entry.get("rand_vec_missing"):
                missing_stats["rand_vec"] += 1
            if "missing_fields" in entry:
                for f in entry["missing_fields"]:
                    if f in missing_stats:
                        missing_stats[f] += 1
        except Exception as e:
            logger.error(f"处理 episode {ep_idx} 时出错: {e}")
            entry = {
                "episode_index": ep_idx,
                "task": task_name,
                "task_index": task_idx,
                "obj_init_pos": None,
                "goal_pose": None,
                "success": False,
                "num_frames": ep_length,
                "missing_fields": ["obj_init_pos", "goal_pose"],
                "error": str(e),
            }
            episodes_list.append(entry)
            missing_stats["obj_init_pos"] += 1
            missing_stats["goal_pose"] += 1

    episodes_list.sort(key=lambda x: x["episode_index"])

    logger.info("")
    logger.info("=" * 60)
    logger.info("恢复统计")
    logger.info("=" * 60)
    logger.info(f"数据集路径: {dataset_path}")
    logger.info(f"发现的 episode 总数: {len(episodes_list)}")
    logger.info(f"各 task 数量统计: {task_counter}")
    logger.info(f"成功恢复环境状态的 episode 数量: {recovered_count}/{len(episodes_list)}")
    logger.info(f"缺失字段统计: {missing_stats}")

    output = {
        "env_state_structure": env_state_structure,
        "num_episodes": len(episodes_list),
        "task_stats": task_counter,
        "episodes": episodes_list,
    }

    if output_path is None:
        output_path = str(dataset_path / "episode_initial_states.json")
    output_path = Path(output_path)

    if dry_run:
        logger.info(f"\n[Dry Run] 将写入 {len(episodes_list)} 个 episode 到 {output_path}")
        return output

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"最终生成 json 路径: {output_path}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("自动验证")
    logger.info("=" * 60)

    with open(output_path, "r") as f:
        saved = json.load(f)

    saved_episodes = saved.get("episodes", [])
    saved_count = len(saved_episodes)
    validation_passed = True

    if saved_count != total_episodes_info:
        logger.error(
            f"验证失败: 生成的 episode 数量 ({saved_count}) "
            f"与 meta/info.json 中的 total_episodes ({total_episodes_info}) 不一致"
        )
        validation_passed = False
    else:
        logger.info(f"[PASS] episode 数量一致: {saved_count} == {total_episodes_info}")

    saved_indices = [ep["episode_index"] for ep in saved_episodes]
    expected_indices = list(range(total_episodes_info))
    if saved_indices != expected_indices:
        missing = set(expected_indices) - set(saved_indices)
        extra = set(saved_indices) - set(expected_indices)
        dupes = [x for x in saved_indices if saved_indices.count(x) > 1]
        dupes = sorted(set(dupes))
        msg_parts = []
        if missing:
            msg_parts.append(f"缺失: {sorted(missing)}")
        if extra:
            msg_parts.append(f"多余: {sorted(extra)}")
        if dupes:
            msg_parts.append(f"重复: {dupes}")
        logger.error(f"验证失败: episode_index 不连续或不完整。{'; '.join(msg_parts)}")
        validation_passed = False
    else:
        logger.info(f"[PASS] episode_index 覆盖 0 到 {total_episodes_info - 1}，无重复")

    saved_tasks = set(ep.get("task") for ep in saved_episodes)
    if len(saved_tasks) != total_tasks_info:
        logger.warning(
            f"[WARN] JSON 中的 task 数量 ({len(saved_tasks)}) "
            f"与 info.json 中的 total_tasks ({total_tasks_info}) 不一致。"
            f"Tasks: {saved_tasks}"
        )
    else:
        logger.info(f"[PASS] task 数量一致: {len(saved_tasks)} == {total_tasks_info}")

    if not validation_passed:
        logger.error("验证未通过，请检查数据集完整性。")
        sys.exit(1)

    logger.info("验证全部通过。")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="根据已有 LeRobot 数据集重新生成 episode_initial_states.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/coffee_push_corner3",
        help="LeRobot 数据集根目录路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 JSON 文件路径 (默认: dataset/episode_initial_states.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印信息，不写入文件",
    )
    args = parser.parse_args()

    rebuild_episodeInitialStates(
        dataset_path=args.dataset,
        output_path=args.output,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()