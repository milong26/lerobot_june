"""
相机视角扰动模块

在MuJoCo仿真器中修改corner2相机的位姿，模拟LIBERO-Plus的camera viewpoint轴。
使用mujoco官方Python绑定（非mujoco-py）。

API:
    env.model.cam_pos[cam_id] = [x, y, z]
    env.model.cam_quat[cam_id] = [w, x, y, z]
"""

import numpy as np
import mujoco
from typing import Dict, Any, Optional


def get_camera_id(model, camera_name: str) -> int:
    """
    获取相机ID

    参数:
        model: mujoco.MjModel实例
        camera_name: 相机名称，如"corner2"

    返回:
        相机ID
    """
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)


def perturb_camera(env, level: str, camera_name: str = "corner2", params: Optional[Dict[str, Any]] = None) -> None:
    """
    扰动相机视角

    在env reset后、render前调用。修改后需要mj_forward重新计算渲染管线。

    参数:
        env: Meta-World环境实例（包含env.model和env.data）
        level: 扰动等级，"L0"表示无扰动
        camera_name: 相机名称，默认"corner2"
        params: 扰动参数字典，来自probe_config.yaml
    """
    if level == "L0" or params is None:
        return

    cam_id = get_camera_id(env.model, camera_name)

    # 获取原始相机位置
    original_pos = env.model.cam_pos[cam_id].copy()

    # 从相机位置计算方位角和仰角（假设相机朝向原点）
    # 方位角：在xy平面上与x轴的夹角
    original_azimuth = np.arctan2(original_pos[1], original_pos[0]) * 180 / np.pi
    # 到原点的水平距离
    original_xy_dist = np.sqrt(original_pos[0]**2 + original_pos[1]**2)
    # 仰角：与xy平面的夹角
    original_elevation = np.arctan2(original_pos[2], original_xy_dist) * 180 / np.pi if original_xy_dist > 0 else 0
    # 到原点的总距离
    original_distance = np.linalg.norm(original_pos)

    # 应用偏移
    azimuth_offset = params.get("azimuth_offset", 0)
    elevation_offset = params.get("elevation_offset", 0)
    distance_offset = params.get("distance_offset", 0)

    new_azimuth = original_azimuth + azimuth_offset
    new_elevation = original_elevation + elevation_offset
    new_distance = original_distance + distance_offset

    # 将新的球坐标转回笛卡尔坐标
    new_xy_dist = new_distance * np.cos(new_elevation * np.pi / 180)
    new_z = new_distance * np.sin(new_elevation * np.pi / 180)

    new_x = new_xy_dist * np.cos(new_azimuth * np.pi / 180)
    new_y = new_xy_dist * np.sin(new_azimuth * np.pi / 180)

    env.model.cam_pos[cam_id] = [new_x, new_y, new_z]

    # 重新计算前向动力学以更新渲染管线
    mujoco.mj_forward(env.model, env.data)


def get_camera_config(level: str) -> Dict[str, Any]:
    """
    获取预定义的相机扰动配置

    参数:
        level: "L0", "L1", 或 "L2"

    返回:
        扰动参数字典
    """
    configs = {
        "L0": None,
        "L1": {
            "azimuth_offset": 5,
            "elevation_offset": 3,
            "distance_offset": 0.1,
        },
        "L2": {
            "azimuth_offset": 10,
            "elevation_offset": 6,
            "distance_offset": 0.2,
        },
    }
    return configs.get(level, None)


if __name__ == "__main__":
    print("相机扰动模块测试")
    print("L0:", get_camera_config("L0"))
    print("L1:", get_camera_config("L1"))
    print("L2:", get_camera_config("L2"))
    print("相机扰动需要在Meta-World环境中测试，请运行run_probe_rollout.py进行集成测试")