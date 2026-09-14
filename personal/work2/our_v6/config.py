"""
V6 AdaptiveGrid with V5 Action Descriptor Configuration

Combines V4's from-scratch acquisition strategy with V5's action descriptor approach.
- Stage 1: coarse uniform coverage (one episode per coarse cell)
- Stage 2: adaptive acquisition based on spatial need + visual disagreement + action disagreement
- Action embeddings use V5's pre-computed action descriptors (causal access maintained)
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

# Random seed
SEED = 42

# Normalization flags
VISUAL_NORMALIZE = True