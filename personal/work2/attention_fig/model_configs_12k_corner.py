"""
Model configurations for 12k checkpoint corner 7-model analysis.

This file defines CORNER_12K_MODEL_CONFIGS for post-hoc attention diagnosis of:
  ours_v1, ours_v2, ours_v3, ours_v4, random, uniform, zero

All models use corner,gripperPOV camera at 12k checkpoint.
eval_task_success and eval_grasp_success are 200-episode final metrics
used ONLY for post-hoc metadata (tables, sorting, correlation observation).
They MUST NOT participate in seed selection, attention computation,
model inference, parameter adjustment, or any selection decision.
"""

CORNER_12K_MODEL_CONFIGS = [
    {
        "name": "ours_v1_corner_12k",
        "method": "ours_v1",
        "camera_name": "corner,gripperPOV",
        "path": "personal/work2/duibi/ours_112_seed42_corner/dynamicanchor_112_seed42/checkpoints/012000/pretrained_model",
        "eval_task_success": 20.0,
        "eval_grasp_success": 90.0,
    },
    {
        "name": "ours_v2_corner_12k",
        "method": "ours_v2",
        "camera_name": "corner,gripperPOV",
        "path": "personal/work2/duibi/ours_v2_112_seed42_corner/dynamicanchor_v2_112_seed42/checkpoints/012000/pretrained_model",
        "eval_task_success": 25.5,
        "eval_grasp_success": 95.5,
    },
    {
        "name": "ours_v3_corner_12k",
        "method": "ours_v3",
        "camera_name": "corner,gripperPOV",
        "path": "personal/work2/duibi/ours_v3_no_action_112_seed42_corner/dynamicanchor_v3_no_action_112_seed42/checkpoints/012000/pretrained_model",
        "eval_task_success": 32.0,
        "eval_grasp_success": 94.5,
    },
    {
        "name": "ours_v4_corner_12k",
        "method": "ours_v4",
        "camera_name": "corner,gripperPOV",
        "path": "personal/work2/duibi/ours_v4_112_seed42_corner/dynamicgrid_v4_112_seed42/checkpoints/012000/pretrained_model",
        "eval_task_success": 34.5,
        "eval_grasp_success": 95.0,
    },
    {
        "name": "random_corner_12k",
        "method": "random",
        "camera_name": "corner,gripperPOV",
        "path": "personal/work2/duibi/random_42_corner/random_112_seed42/checkpoints/012000/pretrained_model",
        "eval_task_success": 32.0,
        "eval_grasp_success": 94.5,
    },
    {
        "name": "uniform_corner_12k",
        "method": "uniform",
        "camera_name": "corner,gripperPOV",
        "path": "personal/work2/duibi/uniform_42_corner/uniform_112_seed42/checkpoints/012000/pretrained_model",
        "eval_task_success": 28.2,
        "eval_grasp_success": 92.5,
    },
    {
        "name": "zero_corner_12k",
        "method": "zero",
        "camera_name": "corner,gripperPOV",
        "path": "personal/work2/duibi/subzerocore_112_seed42_corner/subzerocore_112_seed42/checkpoints/012000/pretrained_model",
        "eval_task_success": 32.0,
        "eval_grasp_success": 95.0,
    },
]