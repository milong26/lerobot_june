"""
V4 DynamicGrid with Action Embedding Configuration

Adaptive grid-based data collection from scratch with spatial + visual + action signals.
Action embeddings are computed online AFTER episode acquisition (causal access).
"""

# Total collection budget (total episodes to collect)
TOTAL_BUDGET = 112

# Initial coarse grid resolution
INITIAL_GRID_X = 7
INITIAL_GRID_Y = 4

# Stage 1 budget (coarse uniform coverage: one episode per coarse cell)
INITIAL_BUDGET = 28  # = 7 * 4

# Cell splitting parameters
SPLIT_X = 2
SPLIT_Y = 2
MAX_DEPTH = 3

# Scoring weights
VISUAL_GLOBAL_WEIGHT = 1.0
VISUAL_WRIST_WEIGHT = 1.0
VISUAL_WEIGHT = 1.0
SPATIAL_WEIGHT = 1.0
ACTION_WEIGHT = 0.5

# Minimum mapping tolerance for episode-to-target matching (meters)
MIN_MAPPING_TOLERANCE = 0.1

# PCA dimensions (must match existing embedding cache)
PCA_DIM = 32

# Action embedding parameters
TEMPORAL_ACTION_STEPS = 16
ACTION_EMBED_DIM = None  # Will be computed from action sequence shape + statistics
ACTION_NORMALIZE = True

# Random seed
SEED = 42

# Normalization flags
VISUAL_NORMALIZE = True