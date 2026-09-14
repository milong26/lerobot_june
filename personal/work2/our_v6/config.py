"""Configuration for our_v6 causal hierarchical demonstration acquisition.

V6 keeps V5's configuration-region allocation idea, but restores the strict
causal information boundary of V4: before an episode is acquired, only reset
configuration metadata may be inspected. Visual/action content is revealed
only after acquisition.
"""

TOTAL_BUDGET = 112
SEED = 42

# Configuration-space partitioning (same dynamic-K idea as V5).
REGION_RATIO = 0.10
MIN_REGIONS = 16
MAX_REGIONS = 128
B0_REGION_RATIO = 0.20

# Acquired-only region priority.
COVERAGE_WEIGHT = 0.50
REGION_VISUAL_WEIGHT = 0.30
REGION_ACTION_WEIGHT = 0.20

# V4-style target-configuration proposal inside the selected V5-style region.
TARGET_PROPOSALS = 64
MAPPING_TOLERANCE = 0.25  # normalized configuration-space distance

# Visual representation variants.
VISUAL_VARIANTS = ("l2", "online_pca")
DEFAULT_VISUAL_VARIANT = "l2"
PCA_DIM = 32
VISUAL_GLOBAL_WEIGHT = 1.0
VISUAL_WRIST_WEIGHT = 1.0

# V5 action descriptor content with V4-style group normalization/weighting.
ACTION_STEPS = 16
ACTION_TEMPORAL_WEIGHT = 1.0
ACTION_STATS_WEIGHT = 0.25
ACTION_PATH_WEIGHT = 0.10
ACTION_FINAL_L2 = True

EPS = 1e-8
