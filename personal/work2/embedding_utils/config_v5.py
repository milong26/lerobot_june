"""
Configuration constants for V5 embedding extraction.

V5 uses a vision-action episode representation that fuses:
- observation.images.top (top camera)
- observation.images.wrist (wrist camera)
- action trajectory

This config is separate from config.py (V1/V2/V3/V4 SmolVLM config).
"""

from pathlib import Path

# V5 embedding version identifier
V5_EMBEDDING_VERSION = "v5"

# Shared embedding root directory (same root as V1-V4)
V5_SHARED_EMBEDDING_ROOT = Path("/data/zhonglinye/jun/lerobot/personal/work2/shared_embeddings")

# Model configuration
V5_MODEL_NAME = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
V5_PROMPT_TEXT = "<image> Pick and place a puck to a goal"

# Frame sampling: number of frames to sample from each episode for visual feature extraction
V5_FRAME_SAMPLE_COUNT = 16

# Action embedding parameters
V5_TEMPORAL_ACTION_STEPS = 16

# PCA configuration
V5_PCA_DIM = 32

# Output directory name segment for V5
V5_OUTPUT_DIR = "v5_vision_action"


def sanitize_name_v5(value: str) -> str:
    """
    Convert a string to a safe directory name.
    """
    safe = value.replace("/", "-").replace("\\", "-").replace(" ", "_").replace("<", "").replace(">", "")
    safe = safe.replace(":", "-").replace("*", "-").replace("?", "-").replace('"', "-")
    safe = safe.replace("|", "-").replace("(", "").replace(")", "")
    return safe


def build_v5_extraction_method_name(pca_dim: int = V5_PCA_DIM) -> str:
    """
    Build a unique extraction method name for V5.
    
    Output example:
    smolvlm2-500m_v5-vision-action_frames16_action16_pca32_v5
    """
    model_short = sanitize_name_v5(V5_MODEL_NAME).lower().replace("huggingfacetb-", "").replace("smolvlm2-500m-video-instruct", "smolvlm2-500m")
    
    return (
        f"{model_short}_"
        f"v5-vision-action_"
        f"frames{V5_FRAME_SAMPLE_COUNT}_"
        f"action{V5_TEMPORAL_ACTION_STEPS}_"
        f"pca{pca_dim}_"
        f"{V5_EMBEDDING_VERSION}"
    )