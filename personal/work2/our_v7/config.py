"""Configuration for paper-faithful Our-V7 acquisition.

V7 keeps the proven V6 representation/execution pipeline, but makes the
selection rule match the ICRA method text:
  * predefined feasible configuration bounds;
  * deterministic farthest-point KMeans initialization;
  * deterministic farthest-point B0 region initialization;
  * p_k = g_k + u_v_norm + u_a_norm.
"""

TOTAL_BUDGET = 112
SEED = 42

REGION_RATIO = 0.10
MIN_REGIONS = 16
MAX_REGIONS = 128
B0_REGION_RATIO = 0.20

VISUAL_VARIANTS = ("l2", "online_pca")
DEFAULT_VISUAL_VARIANT = "l2"
PCA_DIM = 32

# The action/visual representation itself is reused from Our-V6 unchanged.
# These values document the representation used by that implementation.
ACTION_STEPS = 16
ACTION_TEMPORAL_WEIGHT = 1.0
ACTION_STATS_WEIGHT = 0.25
ACTION_PATH_WEIGHT = 0.10

# Shared acquired-gated raw VLM cache for the two V7 visual variants.
RAW_VISUAL_CACHE_VERSION = "smolvlm_raw_first5_wrist20_70_v1"

EPS = 1e-8
