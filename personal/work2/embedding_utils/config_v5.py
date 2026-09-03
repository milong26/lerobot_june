"""
Configuration constants for V5 episode selection.

V5 uses existing visual embeddings (global + wrist) from V1-V4 pipeline,
plus action descriptor as auxiliary selection signal.

This config is separate from config.py (V1/V2/V3/V4 SmolVLM config).
"""

from pathlib import Path

V5_EMBEDDING_VERSION = "v5"

V5_SHARED_EMBEDDING_ROOT = Path("/data/zhonglinye/jun/lerobot/personal/work2/shared_embeddings")

V5_ACTION_DESCRIPTOR_VERSION = "v1"

V5_ACTION_STEPS = 16

V5_ACTION_FEATURES = "full_stats"

V5_ACTION_DESCRIPTOR_OUTPUT_DIR = Path("/data/zhonglinye/jun/lerobot/personal/work2/action_descriptors")


def sanitize_name_v5(value: str) -> str:
    safe = value.replace("/", "-").replace("\\", "-").replace(" ", "_").replace("<", "").replace(">", "")
    safe = safe.replace(":", "-").replace("*", "-").replace("?", "-").replace('"', "-")
    safe = safe.replace("|", "-").replace("(", "").replace(")", "")
    return safe


def build_v5_action_descriptor_name(
    version: str = V5_ACTION_DESCRIPTOR_VERSION,
    num_steps: int = V5_ACTION_STEPS,
    features: str = V5_ACTION_FEATURES,
) -> str:
    return f"action_descriptor_{version}_steps{num_steps}_{features}"