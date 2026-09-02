"""
V3 No-Action Configuration

Adaptive grid-based data collection simulating real robot acquisition from scratch.
No action embeddings used; decisions based on spatial coverage + visual disagreement only.
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

# Minimum mapping tolerance for episode-to-target matching (meters)
MIN_MAPPING_TOLERANCE = 0.1

# PCA dimensions (must match existing embedding cache)
PCA_DIM = 32

# Random seed
SEED = 42

# Normalization flags
VISUAL_NORMALIZE = True