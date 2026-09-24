"""Smoke tests for MiniVLA LeRobot integration."""

from lerobot.policies import (
    MiniVLAConfig,
    MiniVLAT2Config,
    MiniVLAWristConfig,
    MiniVLAWristPretrainedConfig,
)
from lerobot.policies.factory import make_policy_config


def test_minivla_config_registration():
    assert isinstance(make_policy_config("minivla"), MiniVLAConfig)
    assert isinstance(make_policy_config("minivla_t2"), MiniVLAT2Config)
    assert isinstance(make_policy_config("minivla_wrist"), MiniVLAWristConfig)
    assert isinstance(make_policy_config("minivla_wrist_pretrained"), MiniVLAWristPretrainedConfig)
