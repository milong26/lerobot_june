"""
SIC V2: 向量化、固定 reference anchor universe 的 SIC 计算

与 V1 (sic.py) 的核心区别：
1. reference anchor universe A 在整个 selection 过程中固定不变
2. dbar_global / dbar_wrist 只计算一次，对所有 candidate 共享
3. 预计算 kernel matrix，避免重复 pairwise distance 计算
4. 使用 incremental sigma 更新，避免每次从头计算
5. 真正的 sequential greedy：每步只选一个 episode，立即更新 sigma
"""

import numpy as np
from typing import Dict, Tuple, List, Optional


def tau(t: int, alpha: float = 1.0) -> float:
    """
    次数软化函数

    tau(t) = alpha * log((t+1)/t)

    注意：当前 episode-level selection 中 repeat_count=1，
    所以 tau(1, alpha) = alpha * log(2) 是唯一会使用的值。
    repeat softening currently inactive for unique episode-level selection.
    """
    if t < 1:
        raise ValueError(f"t must be >= 1, got {t}")
    return alpha * np.log((t + 1) / t)


def compute_dbar_from_embeddings(
    phi_globals: np.ndarray,
    phi_wrists: np.ndarray
) -> Tuple[float, float, bool]:
    """
    从 embedding 数组计算 dbar_global 和 dbar_wrist

    delta_v(a) = min_{a'!=a} ||phi_v[a] - phi_v[a']||_2
    dbar_v = (1/|A|) * sum_a delta_v(a)

    参数:
        phi_globals: shape (N, D_global)
        phi_wrists: shape (N, D_wrist)

    返回:
        (dbar_global, dbar_wrist, dbar_fallback_used)
    """
    n = len(phi_globals)
    if n < 2:
        return 1.0, 1.0, True

    d_global_list = []
    d_wrist_list = []

    for i in range(n):
        dists_global = np.linalg.norm(phi_globals - phi_globals[i], axis=1)
        dists_wrist = np.linalg.norm(phi_wrists - phi_wrists[i], axis=1)

        dists_global[i] = np.inf
        dists_wrist[i] = np.inf

        d_global_list.append(dists_global.min())
        d_wrist_list.append(dists_wrist.min())

    dbar_global = float(np.mean(d_global_list))
    dbar_wrist = float(np.mean(d_wrist_list))

    dbar_fallback_used = False
    if dbar_global < 1e-6:
        dbar_global = 1.0
        dbar_fallback_used = True
    if dbar_wrist < 1e-6:
        dbar_wrist = 1.0
        dbar_fallback_used = True

    return dbar_global, dbar_wrist, dbar_fallback_used


def build_kernel_matrices(
    phi_globals: np.ndarray,
    phi_wrists: np.ndarray,
    dbar_global: float,
    dbar_wrist: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    预计算 kernel matrix

    K_global[a, c] = exp(-||phi_global[a] - phi_global[c]|| / dbar_global)
    K_wrist[a, c] = exp(-||phi_wrist[a] - phi_wrist[c]|| / dbar_wrist)

    参数:
        phi_globals: shape (N, D_global)
        phi_wrists: shape (N, D_wrist)
        dbar_global: 距离尺度参数
        dbar_wrist: 距离尺度参数

    返回:
        K_global: shape (N, N)
        K_wrist: shape (N, N)
    """
    n = len(phi_globals)

    diff_global = phi_globals[:, np.newaxis, :] - phi_globals[np.newaxis, :, :]
    dist_global = np.linalg.norm(diff_global, axis=2)
    K_global = np.exp(-dist_global / dbar_global)

    diff_wrist = phi_wrists[:, np.newaxis, :] - phi_wrists[np.newaxis, :, :]
    dist_wrist = np.linalg.norm(diff_wrist, axis=2)
    K_wrist = np.exp(-dist_wrist / dbar_wrist)

    return K_global, K_wrist


def check_embeddings_valid(
    phi_globals: np.ndarray,
    phi_wrists: np.ndarray,
    near_dup_cosine_threshold: float = 0.99999
) -> Dict:
    """
    检查 embedding 是否有 NaN / Inf / zero-norm / duplicate

    返回:
        {
            "valid": bool,
            "errors": [],
            "warnings": [],
            "stats": {
                "n_episodes": int,
                "global_dim": int,
                "wrist_dim": int,
                "n_exact_dup_global": int,
                "n_exact_dup_wrist": int,
                "n_near_dup_global": int,
                "n_near_dup_wrist": int,
                "zero_norm_global": int,
                "zero_norm_wrist": int,
            }
        }
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {
            "n_episodes": len(phi_globals),
            "global_dim": phi_globals.shape[1],
            "wrist_dim": phi_wrists.shape[1],
            "n_exact_dup_global": 0,
            "n_exact_dup_wrist": 0,
            "n_near_dup_global": 0,
            "n_near_dup_wrist": 0,
            "zero_norm_global": 0,
            "zero_norm_wrist": 0,
        }
    }

    if np.any(np.isnan(phi_globals)):
        result["valid"] = False
        result["errors"].append("phi_globals contains NaN")
    if np.any(np.isnan(phi_wrists)):
        result["valid"] = False
        result["errors"].append("phi_wrists contains NaN")
    if np.any(np.isinf(phi_globals)):
        result["valid"] = False
        result["errors"].append("phi_globals contains Inf")
    if np.any(np.isinf(phi_wrists)):
        result["valid"] = False
        result["errors"].append("phi_wrists contains Inf")

    n = len(phi_globals)
    if n == 0:
        result["valid"] = False
        result["errors"].append("No embeddings provided")
        return result

    norms_global = np.linalg.norm(phi_globals, axis=1)
    norms_wrist = np.linalg.norm(phi_wrists, axis=1)

    zero_g = int(np.sum(norms_global < 1e-10))
    zero_w = int(np.sum(norms_wrist < 1e-10))
    result["stats"]["zero_norm_global"] = zero_g
    result["stats"]["zero_norm_wrist"] = zero_w

    if zero_g > 0:
        result["valid"] = False
        result["errors"].append(f"{zero_g} phi_globals have zero norm")
    if zero_w > 0:
        result["valid"] = False
        result["errors"].append(f"{zero_w} phi_wrists have zero norm")

    def count_exact_duplicates(phi):
        # 使用 np.unique 进行精确重复检测
        # 返回的是 exact duplicate groups 数量（即出现次数>1的unique vector group数量）
        unique_arr, counts = np.unique(phi, axis=0, return_counts=True)
        return int(np.sum(counts > 1))

    def count_near_duplicate_pairs(phi, threshold):
        # 统计 near duplicate pair 数量，排除 exact duplicate
        # 使用 np.unique 获取 exact duplicate group id
        unique_arr, inverse_indices = np.unique(phi, axis=0, return_inverse=True)
        
        # 归一化后计算 cosine similarity
        norms = np.linalg.norm(phi, axis=1)
        if norms.min() < 1e-10:
            return 0
        phi_norm = phi / norms[:, None]
        cos_sim = phi_norm @ phi_norm.T
        
        n = len(phi)
        near_dup_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                # 排除 exact duplicate（同一 group）
                if inverse_indices[i] == inverse_indices[j]:
                    continue
                if cos_sim[i, j] > threshold:
                    near_dup_pairs += 1
        return near_dup_pairs

    result["stats"]["n_exact_dup_global"] = count_exact_duplicates(phi_globals)
    result["stats"]["n_exact_dup_wrist"] = count_exact_duplicates(phi_wrists)
    result["stats"]["n_exact_duplicate_groups_global"] = result["stats"]["n_exact_dup_global"]
    result["stats"]["n_exact_duplicate_groups_wrist"] = result["stats"]["n_exact_dup_wrist"]

    if result["stats"]["n_exact_dup_global"] > 0:
        result["warnings"].append(f"{result['stats']['n_exact_dup_global']} exact duplicate groups in phi_globals")
    if result["stats"]["n_exact_dup_wrist"] > 0:
        result["warnings"].append(f"{result['stats']['n_exact_dup_wrist']} exact duplicate groups in phi_wrists")

    if norms_global.min() > 1e-10 and norms_wrist.min() > 1e-10:
        near_dup_pairs_g = count_near_duplicate_pairs(phi_globals, near_dup_cosine_threshold)
        near_dup_pairs_w = count_near_duplicate_pairs(phi_wrists, near_dup_cosine_threshold)

        result["stats"]["n_near_dup_global"] = near_dup_pairs_g
        result["stats"]["n_near_dup_wrist"] = near_dup_pairs_w
        result["stats"]["n_near_duplicate_pairs_global"] = near_dup_pairs_g
        result["stats"]["n_near_duplicate_pairs_wrist"] = near_dup_pairs_w

        if near_dup_pairs_g > 0 or near_dup_pairs_w > 0:
            result["warnings"].append(
                f"Near-duplicate embedding pairs detected (excluding exact duplicates): "
                f"global={near_dup_pairs_g}, "
                f"wrist={near_dup_pairs_w}"
            )

    return result


class FixedAnchorSIC:
    """
    固定 reference anchor universe 下的 SIC 计算器

    核心 invariant：
    - reference anchor set A 从创建后永不改变
    - dbar_global / dbar_wrist 从创建后永不改变
    - kernel matrix 预计算后直接查询
    - sigma 向量支持 incremental update

    使用方式：
    1. 用所有 episode embeddings 创建实例
    2. 用 B0 episodes 初始化 selected set
    3. 循环调用 score_candidates() 获取 marginal gains
    4. 调用 select_episode() 正式加入一个 episode
    """

    def __init__(
        self,
        episode_indices: List[int],
        phi_globals: np.ndarray,
        phi_wrists: np.ndarray,
        alpha: float = 1.0,
        lambda_wrist: float = 1.0
    ):
        """
        参数:
            episode_indices: 所有 episode 的索引列表，长度 N
            phi_globals: shape (N, D_global)
            phi_wrists: shape (N, D_wrist)
            alpha: 次数软化系数
            lambda_wrist: wrist 权重
        """
        validation_result = check_embeddings_valid(phi_globals, phi_wrists)
        if not validation_result["valid"]:
            raise ValueError(f"Invalid embeddings: {validation_result['errors']}")

        self.episode_indices = list(episode_indices)
        self.n_episodes = len(episode_indices)
        self.alpha = alpha
        self.lambda_wrist = lambda_wrist

        self.episode_to_idx = {ep: i for i, ep in enumerate(episode_indices)}

        self.phi_globals = phi_globals.copy()
        self.phi_wrists = phi_wrists.copy()

        self.dbar_global, self.dbar_wrist, self.dbar_fallback_used = \
            compute_dbar_from_embeddings(self.phi_globals, self.phi_wrists)

        self.K_global, self.K_wrist = build_kernel_matrices(
            self.phi_globals, self.phi_wrists,
            self.dbar_global, self.dbar_wrist
        )

        self.tau_1 = tau(1, alpha)

        self.sigma_global = np.zeros(self.n_episodes)
        self.sigma_wrist = np.zeros(self.n_episodes)

        self.selected_indices: set = set()

        self._current_sic = 0.0

    @property
    def reference_anchor_count(self) -> int:
        return self.n_episodes

    def _sat(self, x: np.ndarray) -> np.ndarray:
        return x / (1.0 + x)

    def _compute_sic(self) -> float:
        sat_g = self._sat(self.sigma_global)
        sat_w = self._sat(self.sigma_wrist)
        return float(np.sum(sat_g) + self.lambda_wrist * np.sum(sat_w))

    def initialize_b0(self, b0_episodes: List[int]):
        """
        用 B0 episodes 初始化 selected set 和 sigma

        参数:
            b0_episodes: B0 episode 索引列表
        """
        self.selected_indices = set()
        self.sigma_global = np.zeros(self.n_episodes)
        self.sigma_wrist = np.zeros(self.n_episodes)

        for ep in b0_episodes:
            if ep in self.episode_to_idx:
                idx = self.episode_to_idx[ep]
                if idx in self.selected_indices:
                    continue  # Skip duplicate episodes
                self.selected_indices.add(idx)
                self.sigma_global += self.tau_1 * self.K_global[:, idx]
                self.sigma_wrist += self.tau_1 * self.K_wrist[:, idx]

        self._current_sic = self._compute_sic()

    def score_candidates(
        self,
        candidate_episodes: List[int]
    ) -> Dict[int, float]:
        """
        计算每个 candidate 的 marginal gain

        Delta(c) = SIC(D union {c}) - SIC(D)
                 = sum(sat(sigma + tau*K[:,c]) - sat(sigma))
                 + lambda_wrist * sum(sat(sigma_w + tau*K_w[:,c]) - sat(sigma_w))

        所有 candidate 在相同的 sigma 下比较。

        参数:
            candidate_episodes: 候选 episode 索引列表

        返回:
            Dict[episode_index, marginal_gain]
        """
        if not candidate_episodes:
            return {}

        candidate_indices = []
        for ep in candidate_episodes:
            if ep in self.episode_to_idx and ep not in self.selected_indices:
                candidate_indices.append(self.episode_to_idx[ep])

        if not candidate_indices:
            return {}

        cand_arr = np.array(candidate_indices)

        delta_sigma_g = self.tau_1 * self.K_global[:, cand_arr]
        delta_sigma_w = self.tau_1 * self.K_wrist[:, cand_arr]

        sat_before_g = self._sat(self.sigma_global)
        sat_before_w = self._sat(self.sigma_wrist)

        sigma_g_new = self.sigma_global[:, np.newaxis] + delta_sigma_g
        sigma_w_new = self.sigma_wrist[:, np.newaxis] + delta_sigma_w

        sat_after_g = self._sat(sigma_g_new)
        sat_after_w = self._sat(sigma_w_new)

        gain_g = np.sum(sat_after_g - sat_before_g[:, np.newaxis], axis=0)
        gain_w = np.sum(sat_after_w - sat_before_w[:, np.newaxis], axis=0)

        marginal_gains = gain_g + self.lambda_wrist * gain_w

        result = {}
        for i, idx in enumerate(candidate_indices):
            ep = self.episode_indices[idx]
            result[ep] = float(marginal_gains[i])

        return result

    def select_episode(self, episode_idx: int) -> Dict:
        """
        正式选择一个 episode 加入 selected set

        参数:
            episode_idx: episode 索引

        返回:
            选择后的诊断信息
        """
        if episode_idx not in self.episode_to_idx:
            raise ValueError(f"Episode {episode_idx} not in universe")
        if episode_idx in self.selected_indices:
            raise ValueError(f"Episode {episode_idx} already selected")

        idx = self.episode_to_idx[episode_idx]

        sic_before = self._current_sic

        self.sigma_global += self.tau_1 * self.K_global[:, idx]
        self.sigma_wrist += self.tau_1 * self.K_wrist[:, idx]
        self.selected_indices.add(idx)

        self._current_sic = self._compute_sic()

        return {
            "episode_index": episode_idx,
            "sic_before": sic_before,
            "sic_after": self._current_sic,
            "marginal_gain": self._current_sic - sic_before
        }

    def get_current_sic(self) -> float:
        return self._current_sic

    def get_selected_episodes(self) -> List[int]:
        return sorted([self.episode_indices[i] for i in self.selected_indices])

    def get_sigma_stats(self) -> Dict:
        sat_g = self._sat(self.sigma_global)
        sat_w = self._sat(self.sigma_wrist)
        return {
            "mean_sigma_global": float(np.mean(self.sigma_global)),
            "mean_sigma_wrist": float(np.mean(self.sigma_wrist)),
            "mean_sat_global": float(np.mean(sat_g)),
            "mean_sat_wrist": float(np.mean(sat_w)),
            "min_sigma_global": float(np.min(self.sigma_global)),
            "max_sigma_global": float(np.max(self.sigma_global)),
        }