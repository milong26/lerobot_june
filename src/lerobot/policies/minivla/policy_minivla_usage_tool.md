# MiniVLA Usage Guide

Install dependencies:

    pip install -e ".[minivla]"

## Train through the standard LeRobot CLI

Single-camera base MiniVLA:

    lerobot-train \
      --policy.type=minivla \
      --dataset.repo_id=<USER>/<DATASET> \
      --batch_size=8 \
      --steps=50000

For a multi-camera dataset, identify the primary camera explicitly:

    lerobot-train \
      --policy.type=minivla \
      --policy.primary_image_key=observation.images.corner \
      --dataset.repo_id=<USER>/<DATASET> \
      --batch_size=8 \
      --steps=50000

Wrist-camera variant:

    lerobot-train \
      --policy.type=minivla_wrist \
      --policy.primary_image_key=observation.images.corner \
      --policy.wrist_image_key=observation.images.gripperPOV \
      --dataset.repo_id=<USER>/<DATASET> \
      --batch_size=8 \
      --steps=50000

Temporal two-frame variant uses --policy.type=minivla_t2. Its dataset delta indices are [-1, 0].

## Evaluate a LeRobot checkpoint

    lerobot-eval \
      --policy.path=<LOCAL_CHECKPOINT_OR_HF_REPO> \
      --env.type=<ENV_TYPE> \
      --env.task=<TASK>

MiniVLAPolicy.from_pretrained supports both local LeRobot checkpoint directories and Hugging Face Hub repositories.

## Official MiniVLA initialization

For backbone-only initialization:

    lerobot-train \
      --policy.type=minivla_wrist_pretrained \
      --policy.official_init_mode=backbone_only \
      --policy.official_pretrained_checkpoint=<OFFICIAL_CHECKPOINT> \
      --policy.primary_image_key=observation.images.corner \
      --policy.wrist_image_key=observation.images.gripperPOV \
      --dataset.repo_id=<USER>/<DATASET>

This mode imports compatible vision/projector/language weights but keeps the action representation compatible with the current LeRobot dataset.

## VQ mode

When using a VQ action tokenizer, set vq_model_path to a VQ checkpoint whose input_dim_w matches the dataset action dimension. MiniVLA validates this before training.
