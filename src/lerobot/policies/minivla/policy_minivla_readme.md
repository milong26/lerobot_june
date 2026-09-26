# MiniVLA Policy

This directory contains the native LeRobot integration of MiniVLA. The policy follows the same lifecycle as built-in policies such as SmolVLA: configuration is registered through PreTrainedConfig, model classes inherit PreTrainedPolicy, preprocessing is provided through the policy processor factory, and LeRobot-format checkpoints use the standard model.safetensors + config.json layout.

## Policy types

- minivla: one current primary image.
- minivla_t2: two temporal frames from the primary camera.
- minivla_wrist: current primary image plus current wrist image.
- minivla_wrist_pretrained: wrist variant initialized from official MiniVLA backbone weights while keeping the LeRobot action interface.

## Dependencies

Install from the repository root:

    pip install -e ".[minivla]"

The extra includes Transformers, TIMM and the VQ dependencies used by MiniVLA.

## LeRobot interfaces

- MiniVLAConfig and registered variants with optimizer/scheduler presets and dataset delta indices.
- MiniVLAPolicy and variant policy classes with forward, predict_action_chunk, select_action, reset, get_optim_params, save_pretrained, and from_pretrained.
- make_minivla_pre_post_processors for LeRobot dataset/environment preprocessing and action postprocessing.
- DINO/SigLIP image transforms, Qwen2.5 language backbone, non-VQ action tokenization, and optional compatible VQ action tokenization.

## Camera resolution

A single-camera base dataset is resolved automatically. Multi-camera datasets should set primary_image_key explicitly. Wrist variants can infer a uniquely named wrist/gripper/hand/EEF camera; otherwise set both camera keys explicitly. Ambiguous mappings fail early instead of silently choosing the wrong camera.

## Checkpoints

Standard LeRobot checkpoints can be loaded from either a local directory or a Hugging Face Hub repository through --policy.path, matching other built-in policies. Official MiniVLA .pt artifacts remain supported as an initialization/compatibility path when an explicit LeRobot config is supplied.

## Action compatibility

The policy validates the dataset action dimension against a VQ tokenizer when VQ mode is enabled. It does not hard-code a robot-specific action dimension. For a new embodiment, use a VQ checkpoint trained for that action dimension or use the non-VQ action tokenizer.
