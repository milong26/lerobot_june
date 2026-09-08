"""
Configuration constants for shared embedding cache.

All embedding extraction parameters are defined here to ensure
consistency across all experiments and the shared cache system.
"""

import json
from pathlib import Path

# Shared embedding root directory
SHARED_EMBEDDING_ROOT = Path("/data/zhonglinye/jun/lerobot/personal/work2/shared_embeddings")

# Model configuration
MODEL_NAME = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

# MetaWorld task descriptions config
METAWORLD_CONFIG_PATH = Path("/data/zhonglinye/jun/lerobot/src/lerobot/envs/metaworld_config.json")

# Default prompt text (fallback when task description not found)
DEFAULT_PROMPT_TEXT = "<image> Pick and place a puck to a goal"

# Token pooling strategy
TOKEN_POOLING = "last_hidden_token_mean"

# Frame selection rules
GLOBAL_FRAME_RULE = "first5"
WRIST_START_RATIO = 0.2
WRIST_END_RATIO = 0.7
TEMPORAL_POOLING = "mean"

# PCA configuration
DEFAULT_PCA_DIM = 32

# Extractor version - increment when extraction logic changes
EXTRACTOR_VERSION = "v1"


def _extract_task_name(dataset_name: str) -> str:
    """
    Extract MetaWorld task name from dataset name.
    
    Examples:
        coffee-button-v3_corner -> coffee-button-v3
        pick_place_corner -> pick-place-v3
        disassemble-v3_corner -> disassemble-v3
        pick_place_v3_top -> pick-place-v3
    """
    # Remove view suffixes: _corner, _top, _gripper, _left, _right, etc.
    import re
    base = re.sub(r'_(corner|top|gripper|left|right|front|back|view)[0-9]*$', '', dataset_name)
    # Convert underscores to hyphens
    base = base.replace('_', '-')
    # Add -v3 if not already present
    if 'v3' not in base and 'v2' not in base:
        base = f'{base}-v3'
    return base


def get_prompt_text(dataset_name: str) -> str:
    """
    Get task-specific prompt text from MetaWorld config.
    
    Args:
        dataset_name: Dataset name like 'coffee-button-v3_corner' or 'pick_place_corner'
    
    Returns:
        Prompt text like '<image> Push a button on the coffee machine'
    """
    task_name = _extract_task_name(dataset_name)
    
    # Try to load from metaworld_config.json
    if METAWORLD_CONFIG_PATH.exists():
        try:
            with open(METAWORLD_CONFIG_PATH) as f:
                config = json.load(f)
            task_descriptions = config.get('TASK_DESCRIPTIONS', {})
            if task_name in task_descriptions:
                return f"<image> {task_descriptions[task_name]}"
        except Exception:
            pass
    
    # Fallback to default
    return DEFAULT_PROMPT_TEXT


# Backward compatibility: keep PROMPT_TEXT as default
PROMPT_TEXT = DEFAULT_PROMPT_TEXT


def sanitize_name(value: str) -> str:
    """
    Convert a string to a safe directory name.
    Replaces problematic characters with safe alternatives.
    """
    safe = value.replace("/", "-").replace("\\", "-").replace(" ", "_").replace("<", "").replace(">", "")
    safe = safe.replace(":", "-").replace("*", "-").replace("?", "-").replace('"', "-")
    safe = safe.replace("|", "-").replace("(", "").replace(")", "")
    return safe


# Explicit mapping for token_pooling to method name segment
TOKEN_POOLING_NAMES = {
    "last_hidden_token_mean": "last-hidden-tokenmean",
}


def build_extraction_method_name(pca_dim: int = DEFAULT_PCA_DIM) -> str:
    """
    Build a unique extraction method name based on all extraction parameters.
    
    This ensures that any change in model, pooling, frame rules, wrist range,
    PCA dimension, or extractor version produces a different directory name.
    
    Default output:
    smolvlm2-500m_last-hidden-tokenmean_global-first5_wrist-20to70_temporal-mean_pca32_v1
    """
    model_short = sanitize_name(MODEL_NAME).lower().replace("huggingfacetb-", "").replace("smolvlm2-500m-video-instruct", "smolvlm2-500m")
    
    # Use explicit mapping for token_pooling, fallback to sanitize
    token_pooling_short = TOKEN_POOLING_NAMES.get(TOKEN_POOLING, sanitize_name(TOKEN_POOLING).replace("_", "-"))
    
    global_rule_short = GLOBAL_FRAME_RULE.replace("_", "-")
    wrist_start_pct = int(WRIST_START_RATIO * 100)
    wrist_end_pct = int(WRIST_END_RATIO * 100)
    temporal_short = TEMPORAL_POOLING
    
    return (
        f"{model_short}_{token_pooling_short}_"
        f"global-{global_rule_short}_"
        f"wrist-{wrist_start_pct}to{wrist_end_pct}_"
        f"temporal-{temporal_short}_"
        f"pca{pca_dim}_{EXTRACTOR_VERSION}"
    )