# MiniVLA Usage Guide

## Import

```python
from lerobot.policies.minivla import MiniVLAPolicy, MiniVLAConfig
```

## Configuration

Configure model behavior through `MiniVLAConfig`.

Important fields:

- `official_init_mode`: initialization strategy
  - `none`
  - `backbone_only`
- `official_pretrained_checkpoint`: optional official MiniVLA checkpoint path
- `action_tokenizer_type`: action representation
- `vq_model_path`: compatible VQ action tokenizer path
- `primary_image_key`: primary camera input
- `wrist_image_key`: wrist camera input when enabled

## Training defaults

The policy provides MiniVLA-oriented defaults:

- AdamW optimizer
- learning rate `2e-5`
- warmup-compatible scheduler configuration
- quantile action/state normalization

Training entry points outside this directory should only pass runtime options and dataset-specific information.

## Dataset adaptation

For a new LeRobot dataset:

1. Set camera keys explicitly.
2. Verify action dimension compatibility.
3. Use a matching VQ tokenizer if VQ mode is enabled.
4. Provide dataset statistics for normalization.

## Debugging

Common issues:

- Action dimension mismatch: use a compatible action tokenizer.
- Missing camera key: configure `primary_image_key` explicitly.
- Incorrect checkpoint loading: verify initialization mode and checkpoint type.
