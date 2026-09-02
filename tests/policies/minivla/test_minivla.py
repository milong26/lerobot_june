import tempfile

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import make_policy_config
from lerobot.policies.minivla.configuration_minivla import MiniVLAConfig
from lerobot.policies.minivla.modeling_minivla import MiniVLAPolicy
from lerobot.policies.minivla.tokenizer import SimpleTokenizer
from lerobot.utils.constants import ACTION, OBS_STATE


def _make_dummy_features(state_dim=39, action_dim=4):
    return {
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        },
        "output_features": {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
        },
    }


def _make_dummy_batch(B=2, state_dim=39, action_dim=4, device="cpu"):
    return {
        "observation.images.top": torch.rand(B, 3, 64, 64, device=device),
        "observation.images.wrist": torch.rand(B, 3, 64, 64, device=device),
        OBS_STATE: torch.rand(B, state_dim, device=device),
        ACTION: torch.rand(B, 1, action_dim, device=device),
        "task": ["pick up the block", "place the block"],
    }


def test_tokenizer_deterministic():
    tok = SimpleTokenizer()
    assert tok.vocab_size == 2
    assert tok.pad_id == 0
    assert tok.unk_id == 1

    tok.build_from_texts(["hello world", "hello there"])
    assert "hello" in tok.vocab
    assert "world" in tok.vocab
    assert tok.vocab_size == 4

    ids1 = tok.encode("hello world")
    ids2 = tok.encode("hello world")
    assert ids1 == ids2

    tok2 = SimpleTokenizer(vocab=tok.vocab)
    ids3 = tok2.encode("hello world")
    assert ids1 == ids3


def test_config_registration():
    known = PreTrainedConfig.get_known_choices()
    assert "minivla" in known

    cfg = make_policy_config("minivla")
    assert isinstance(cfg, MiniVLAConfig)


def test_feature_inference():
    feats = _make_dummy_features()
    cfg = MiniVLAConfig(**feats)
    cfg.validate_features()
    assert cfg.state_dim == 39
    assert cfg.action_dim == 4
    assert cfg.image_key == "observation.images.top"


def test_forward_backward():
    feats = _make_dummy_features()
    cfg = MiniVLAConfig(**feats, diffusion_T=2, device="cpu")
    cfg.validate_features()

    policy = MiniVLAPolicy(cfg)
    batch = _make_dummy_batch(state_dim=39, action_dim=4, device="cpu")

    loss, _ = policy(batch)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss > 0

    loss.backward()
    for name, p in policy.model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


def test_action_shapes():
    feats = _make_dummy_features()
    cfg = MiniVLAConfig(**feats, diffusion_T=2, device="cpu")
    cfg.validate_features()

    policy = MiniVLAPolicy(cfg)
    batch = _make_dummy_batch(state_dim=39, action_dim=4, device="cpu")

    chunk = policy.predict_action_chunk(batch)
    assert chunk.shape == (2, 1, 4)

    action = policy.select_action(batch)
    assert action.shape == (2, 4)


def test_multi_camera_only_top():
    feats = _make_dummy_features()
    cfg = MiniVLAConfig(**feats, diffusion_T=2, device="cpu")
    cfg.validate_features()

    policy = MiniVLAPolicy(cfg)
    batch = _make_dummy_batch(state_dim=39, action_dim=4, device="cpu")

    loss, _ = policy(batch)
    assert torch.isfinite(loss)


def test_fallback_single_camera():
    feats = {
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(39,)),
            "observation.images.left": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        },
        "output_features": {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
        },
    }
    cfg = MiniVLAConfig(**feats, diffusion_T=2, device="cpu")
    cfg.validate_features()
    assert cfg.image_key == "observation.images.left"

    policy = MiniVLAPolicy(cfg)
    batch = {
        OBS_STATE: torch.rand(2, 39, device="cpu"),
        "observation.images.left": torch.rand(2, 3, 64, 64, device="cpu"),
        ACTION: torch.rand(2, 1, 4, device="cpu"),
        "task": ["test task"],
    }
    loss, _ = policy(batch)
    assert torch.isfinite(loss)


def test_task_index_fallback():
    feats = _make_dummy_features()
    cfg = MiniVLAConfig(
        **feats,
        diffusion_T=2,
        device="cpu",
        task_texts=["pick up", "place down"],
        default_instruction="pick up",
    )
    cfg.validate_features()

    policy = MiniVLAPolicy(cfg)
    batch = {
        "observation.images.top": torch.rand(2, 3, 64, 64, device="cpu"),
        OBS_STATE: torch.rand(2, 39, device="cpu"),
        ACTION: torch.rand(2, 1, 4, device="cpu"),
        "task_index": torch.tensor([0, 1]),
    }
    loss, _ = policy(batch)
    assert torch.isfinite(loss)


def test_missing_task_uses_default():
    feats = _make_dummy_features()
    cfg = MiniVLAConfig(
        **feats,
        diffusion_T=2,
        device="cpu",
        default_instruction="default instruction",
    )
    cfg.validate_features()

    policy = MiniVLAPolicy(cfg)
    batch = {
        "observation.images.top": torch.rand(2, 3, 64, 64, device="cpu"),
        OBS_STATE: torch.rand(2, 39, device="cpu"),
        ACTION: torch.rand(2, 1, 4, device="cpu"),
    }
    loss, _ = policy(batch)
    assert torch.isfinite(loss)


def test_save_load_vocab_consistency():
    feats = _make_dummy_features()
    vocab = {"<pad>": 0, "<unk>": 1, "pick": 2, "up": 3, "the": 4, "block": 5}
    cfg = MiniVLAConfig(
        **feats,
        diffusion_T=2,
        device="cpu",
        task_texts=["pick up the block"],
        vocab=vocab,
    )
    cfg.validate_features()

    policy = MiniVLAPolicy(cfg)
    batch = _make_dummy_batch(state_dim=39, action_dim=4, device="cpu")

    action_before = policy.select_action(batch)

    with tempfile.TemporaryDirectory() as tmpdir:
        policy.save_pretrained(tmpdir)
        loaded_policy = MiniVLAPolicy.from_pretrained(tmpdir)

    vocab_before = policy.config.vocab
    vocab_after = loaded_policy.config.vocab
    assert vocab_before == vocab_after

    tok = SimpleTokenizer(vocab=vocab_before)
    tok_loaded = SimpleTokenizer(vocab=vocab_after)
    ids1 = tok.encode("pick up the block")
    ids2 = tok_loaded.encode("pick up the block")
    assert ids1 == ids2

    action_after = loaded_policy.select_action(batch)
    assert action_after.shape == action_before.shape