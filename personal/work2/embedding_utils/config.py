"""
Configuration constants for shared embedding cache.

All embedding extraction parameters are defined here to ensure
consistency across all experiments and the shared cache system.
"""

from pathlib import Path

# Shared embedding root directory
SHARED_EMBEDDING_ROOT = Path("/data/zhonglinye/jun/lerobot/personal/work2/shared_embeddings")

# Model configuration
MODEL_NAME = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
PROMPT_TEXT = "<image> Pick and place a puck to a goal"

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