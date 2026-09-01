"""
Dynamic Anchor v2 Configuration

Default parameters for v2 episode selection.
"""

# Target subset size
TARGET_SIZE = 112

# Random seed
SEED = 42

# Seed set size for initialization
SEED_SIZE = 18

# kNN density parameter
KNN_K = 5

# Embedding dimensions (after PCA)
VISUAL_EMBEDDING_DIM = 32  # per view (global + wrist)

# Action embedding switch
USE_ACTION_EMBEDDING = True

# Action embedding extraction method
ACTION_EMBEDDING_METHOD = "combined"

# Embedding weights for fusion
ACTION_WEIGHT = 1.0
VISUAL_WEIGHT = 1.0

# Normalization flags
ACTION_NORMALIZE = True
VISUAL_NORMALIZE = True