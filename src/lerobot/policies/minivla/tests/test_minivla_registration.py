"""Smoke tests for MiniVLA LeRobot integration."""

from lerobot.policies import (
    MiniVLAConfig,
    MiniVLAT2Config,
    MiniVLAWristConfig,
    MiniVLAWristPretrainedConfig,
)
from lerobot.policies.factory import get_policy_class, make_policy_config


def test_minivla_config_registration():
    assert isinstance(make_policy_config("minivla"), MiniVLAConfig)
    assert isinstance(make_policy_config("minivla_t2"), MiniVLAT2Config)
    assert isinstance(make_policy_config("minivla_wrist"), MiniVLAWristConfig)
    assert isinstance(make_policy_config("minivla_wrist_pretrained"), MiniVLAWristPretrainedConfig)


def test_minivla_policy_class_resolution():
    expected = {
        "minivla": "MiniVLAPolicy",
        "minivla_t2": "MiniVLAT2Policy",
        "minivla_wrist": "MiniVLAWristPolicy",
        "minivla_wrist_pretrained": "MiniVLAWristPretrainedPolicy",
    }
    for policy_type, class_name in expected.items():
        assert get_policy_class(policy_type).__name__ == class_name


def test_minivla_action_step_validation():
    try:
        MiniVLAConfig(chunk_size=1, n_action_steps=2)
    except ValueError as exc:
        assert "cannot exceed chunk_size" in str(exc)
    else:
        raise AssertionError("MiniVLAConfig must reject n_action_steps > chunk_size")
