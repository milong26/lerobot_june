"""
DemInf Isolation Tests

Tests to verify that DemInf module runs independently without
dependencies on Ours, embedding, VLM, visual feature modules.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_deminf_import_isolation():
    """
    Check that the running environment does not have ours, embedding, vlm dependencies.

    DemInf must not import or depend on these modules.
    """
    import importlib

    forbidden_modules = [
        "ours", "embedding", "vlm", "clip", "visual_feature",
        "personal.work2.ours", "personal.work2.embedding",
        "personal.work2.vlm", "personal.work2.clip",
        "personal.work2.visual_feature",
    ]

    for mod_name in forbidden_modules:
        try:
            importlib.import_module(mod_name)
            pytest.fail(
                f"Forbidden module '{mod_name}' is importable. "
                f"DemInf must not depend on this module."
            )
        except (ImportError, ModuleNotFoundError):
            pass

    from deminf.config import DemInfConfig
    from deminf.score_episodes import score_latents
    from deminf.select_subset import select_top_episodes

    assert DemInfConfig is not None
    assert score_latents is not None
    assert select_top_episodes is not None


def test_deminf_config_reject_visual_feature():
    """
    When visual-related configuration is provided, validate_deminf_config must raise RuntimeError.
    """
    from deminf.config import DemInfConfig, validate_deminf_config

    config = DemInfConfig(
        dataset_path="/tmp/test_dataset",
        output_dir="/tmp/test_output",
    )

    config.visual_embedding_path = "/some/path"
    with pytest.raises(RuntimeError, match="visual_embedding_path"):
        validate_deminf_config(config)

    config2 = DemInfConfig(
        dataset_path="/tmp/test_dataset",
        output_dir="/tmp/test_output",
    )
    config2.vlm_checkpoint = "/some/checkpoint"
    with pytest.raises(RuntimeError, match="vlm_checkpoint"):
        validate_deminf_config(config2)

    config3 = DemInfConfig(
        dataset_path="/tmp/test_dataset",
        output_dir="/tmp/test_output",
    )
    config3.clip_model = "ViT-B/32"
    with pytest.raises(RuntimeError, match="clip_model"):
        validate_deminf_config(config3)

    config4 = DemInfConfig(
        dataset_path="/tmp/test_dataset",
        output_dir="/tmp/test_output",
    )
    config4.image_feature = "some_feature"
    with pytest.raises(RuntimeError, match="image_feature"):
        validate_deminf_config(config4)


def test_deminf_config_reject_other_scores():
    """
    When non-DemInf scoring parameters are provided, validate_deminf_config must raise RuntimeError.
    """
    from deminf.config import DemInfConfig, validate_deminf_config

    config = DemInfConfig(
        dataset_path="/tmp/test_dataset",
        output_dir="/tmp/test_output",
    )
    config.ours_score = 0.5
    with pytest.raises(RuntimeError, match="ours_score"):
        validate_deminf_config(config)

    config2 = DemInfConfig(
        dataset_path="/tmp/test_dataset",
        output_dir="/tmp/test_output",
    )
    config2.sic_score = 0.3
    with pytest.raises(RuntimeError, match="sic_score"):
        validate_deminf_config(config2)

    config3 = DemInfConfig(
        dataset_path="/tmp/test_dataset",
        output_dir="/tmp/test_output",
    )
    config3.embedding_score = 0.7
    with pytest.raises(RuntimeError, match="embedding_score"):
        validate_deminf_config(config3)


def test_deminf_config_reject_wrong_representation_type():
    """
    When representation_type is not 'state_action', validate_deminf_config must raise RuntimeError.
    """
    from deminf.config import DemInfConfig, validate_deminf_config

    config = DemInfConfig(
        dataset_path="/tmp/test_dataset",
        output_dir="/tmp/test_output",
        representation_type="image",
    )
    with pytest.raises(RuntimeError, match="representation_type"):
        validate_deminf_config(config)

    config2 = DemInfConfig(
        dataset_path="/tmp/test_dataset",
        output_dir="/tmp/test_output",
        representation_type="state",
    )
    with pytest.raises(RuntimeError, match="representation_type"):
        validate_deminf_config(config2)


def test_deminf_config_reject_wrong_state_source():
    """
    When state_source is not 'observation.environment_state', validate_deminf_config must raise RuntimeError.
    """
    from deminf.config import DemInfConfig, validate_deminf_config

    config = DemInfConfig(
        dataset_path="/tmp/test_dataset",
        output_dir="/tmp/test_output",
        state_source="observation.state",
    )
    with pytest.raises(RuntimeError, match="state_source"):
        validate_deminf_config(config)


def test_subset_metadata_contains_deminf_fields():
    """
    Check that subset JSON contains algorithm, representation, uses_reward,
    uses_policy_rollout fields.
    """
    import pandas as pd
    from deminf.config import DemInfConfig
    from deminf.select_subset import save_subset_json

    with tempfile.TemporaryDirectory() as tmpdir:
        config = DemInfConfig(
            dataset_path="/tmp/test_dataset",
            output_dir=tmpdir,
        )

        score_table = pd.DataFrame({
            "episode_idx": [0, 1, 2],
            "deminf_score": [1.0, 0.5, 0.0],
            "rank": [1, 2, 3],
        })

        output_path = Path(tmpdir) / "test_subset.json"
        subset_data = save_subset_json(
            selected_indices=[0, 1],
            score_table=score_table,
            config=config,
            output_path=output_path,
            relative_action=True,
            state_dim=39,
            action_dim=4,
        )

        params = subset_data["parameters"]

        assert "algorithm" in params
        assert params["algorithm"] == "DemInf"

        assert "representation" in params
        assert params["representation"] == "state_action"

        assert "uses_reward" in params
        assert params["uses_reward"] is False

        assert "uses_policy_rollout" in params
        assert params["uses_policy_rollout"] is False

        assert "uses_visual_embedding" in params
        assert params["uses_visual_embedding"] is False

        assert "score_type" in params
        assert params["score_type"] == "ksg_mutual_information"

        assert "state_key" in params
        assert params["state_key"] == "observation.environment_state"

        assert "action_key" in params
        assert params["action_key"] == "action"


def test_score_latents_isolation():
    """
    Verify that score_latents() only operates on latents and KSG,
    without any VAE models or raw data.
    """
    import pandas as pd
    from deminf.config import DemInfConfig
    from deminf.score_episodes import score_latents

    with tempfile.TemporaryDirectory() as tmpdir:
        config = DemInfConfig(
            dataset_path="/tmp/test_dataset",
            output_dir=tmpdir,
            quality_batch_size=128,
            quality_repeat=2,
            ks=(5,),
        )

        n = 256
        z_s = np.random.randn(n, 12).astype(np.float32)
        z_a = np.random.randn(n, 6).astype(np.float32)
        episode_ids = np.repeat(np.arange(8), n // 8)
        timestep_ids = np.tile(np.arange(n // 8), 8)
        global_row_ids = np.arange(n)

        ep_scores, ts_scores = score_latents(
            z_s=z_s,
            z_a=z_a,
            episode_ids=episode_ids,
            timestep_ids=timestep_ids,
            global_row_ids=global_row_ids,
            config=config,
            output_dir=tmpdir,
        )

        assert isinstance(ep_scores, pd.DataFrame)
        assert isinstance(ts_scores, pd.DataFrame)
        assert "deminf_score" in ep_scores.columns
        assert "rank" in ep_scores.columns
        assert len(ep_scores) == 8

        ep_csv = Path(tmpdir) / "episode_scores.csv"
        ts_csv = Path(tmpdir) / "timestep_scores.csv"
        assert ep_csv.exists()
        assert ts_csv.exists()

        ep_df = pd.read_csv(str(ep_csv))
        assert "episode_id" in ep_df.columns
        assert "deminf_score" in ep_df.columns
        assert "rank" in ep_df.columns

        ts_df = pd.read_csv(str(ts_csv))
        assert "episode_id" in ts_df.columns
        assert "timestep_id" in ts_df.columns
        assert "raw_ksg_score" in ts_df.columns
        assert "normalized_score" in ts_df.columns
        assert "repeat_id" in ts_df.columns
        assert "batch_id" in ts_df.columns