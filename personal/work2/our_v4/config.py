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

# Action embedding internal feature group weights
# These control the relative contribution of each feature group INSIDE the action embedding.
# They are distinct from ACTION_WEIGHT, which controls how much the entire action
# disagreement signal contributes to the final cell priority.
#
# Final ActionEmbedding formula:
#   ActionEmb = [
#       ACTION_TEMPORAL_WEIGHT * L2_norm(temporal_flat),
#       ACTION_STATS_WEIGHT  * L2_norm(statistics_vector),
#       ACTION_LENGTH_WEIGHT * scaled_length
#   ]
# The temporal part is the PRIMARY signal; statistics and length are auxiliary.
ACTION_TEMPORAL_WEIGHT = 1.0
ACTION_STATS_WEIGHT = 0.25
ACTION_LENGTH_WEIGHT = 0.1

# Length scaling factor for stable trajectory length representation.
# Uses np.log1p(T) / ACTION_LENGTH_SCALE to avoid T dominating the embedding.
ACTION_LENGTH_SCALE = 10.0

# Random seed
SEED = 42

# Normalization flags
VISUAL_NORMALIZE = True