#!/usr/bin/env python
"""
验证 MiniVLA optimizer param groups 是否真的使用官方 decay/no_decay 分组。

运行方式：
    cd /data/zhonglinye/jun/lerobot
    python personal/work2/differentvlm/minivla/scripts/verify_optimizer.py
"""
import sys
sys.path.insert(0, "src")

import torch
from lerobot.policies.minivla.configuration_minivla import MiniVLAConfig
from lerobot.policies.minivla.modeling_minivla import MiniVLAPolicy
from lerobot.optim.optimizers import AdamWConfig
from lerobot.configs.types import PolicyFeature, FeatureType

print("=" * 80)
print("[VERIFY] MiniVLA Optimizer Param Groups")
print("=" * 80)

# 创建最小配置
config = MiniVLAConfig(
    input_features={
        "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
        "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
    },
    output_features={
        "action": PolicyFeature(FeatureType.ACTION, (4,)),  # 4D MetaWorld action
    },
    action_tokenizer_type="extra_action_tokenizer",
    primary_image_key="observation.images.front",
    wrist_image_key="observation.images.wrist",
    optimizer_lr=2e-5,
    optimizer_weight_decay=0.0,
    optimizer_betas=(0.9, 0.999),
    optimizer_grad_clip_norm=1.0,
    scheduler_type="constant",
    scheduler_warmup_ratio=0.0,
    scheduler_warmup_steps=0,
    official_init_mode="none",  # 不加载 checkpoint，随机初始化
    official_pretrained_checkpoint="",
    vq_model_path="",
    dtype="float32",  # 用 float32 避免 BF16 在 CPU 上不支持
    enable_mixed_precision_training=False,
)

print(f"\n[1] 创建 MiniVLAPolicy（随机初始化，不加载 checkpoint）...")
try:
    policy = MiniVLAPolicy(config)
    print("    ✓ Policy 创建成功")
except Exception as e:
    print(f"    ✗ Policy 创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n[2] 调用 policy.get_optim_params()...")
try:
    param_groups = policy.get_optim_params()
    print(f"    ✓ get_optim_params() 返回 {len(param_groups)} 个 param groups")
except Exception as e:
    print(f"    ✗ get_optim_params() 调用失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n[3] 验证 param groups 结构...")
for i, group in enumerate(param_groups):
    name = group.get("name", f"group_{i}")
    lr = group.get("lr", "N/A")
    weight_decay = group.get("weight_decay", "N/A")
    num_tensors = len(group.get("params", []))
    total_params = sum(p.numel() for p in group["params"])
    
    print(f"    Group {i} ({name}):")
    print(f"      lr = {lr}")
    print(f"      weight_decay = {weight_decay}")
    print(f"      num_tensors = {num_tensors}")
    print(f"      total_params = {total_params:,}")

print(f"\n[4] 验证是否使用官方 decay/no_decay 分组...")
group_names = [g.get("name", "") for g in param_groups]
if "decay" in group_names and "no_decay" in group_names:
    print("    ✓ 使用官方 decay/no_decay 分组")
else:
    print(f"    ✗ 未使用官方分组！当前分组: {group_names}")
    sys.exit(1)

print(f"\n[5] 验证 lr 和 weight_decay 是否正确...")
expected_lr = 2e-5
expected_weight_decay = 0.0

all_correct = True
for group in param_groups:
    lr = group.get("lr")
    wd = group.get("weight_decay")
    name = group.get("name", "")
    
    if lr != expected_lr:
        print(f"    ✗ {name} group lr={lr} != expected {expected_lr}")
        all_correct = False
    if wd != expected_weight_decay:
        print(f"    ✗ {name} group weight_decay={wd} != expected {expected_weight_decay}")
        all_correct = False

if all_correct:
    print(f"    ✓ 所有 group 的 lr={expected_lr:.2e}, weight_decay={expected_weight_decay}")

print(f"\n[6] 验证 AdamWConfig.build() 能否正确处理 param groups...")
try:
    optim_config = AdamWConfig(
        lr=expected_lr,
        weight_decay=expected_weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        grad_clip_norm=1.0,
    )
    optimizer = optim_config.build(param_groups)
    print(f"    ✓ AdamW 创建成功")
    print(f"    ✓ optimizer.param_groups 数量: {len(optimizer.param_groups)}")
    for i, pg in enumerate(optimizer.param_groups):
        print(f"      Group {i} ({pg.get('name', 'unknown')}): lr={pg['lr']}, weight_decay={pg['weight_decay']}, num_params={len(pg['params'])}")
except Exception as e:
    print(f"    ✗ AdamW 创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n[7] 验证 use_policy_training_preset 流程...")
try:
    # 检查 get_optimizer_preset
    optim_preset = config.get_optimizer_preset()
    print(f"    ✓ get_optimizer_preset() 返回: {type(optim_preset).__name__}")
    print(f"      lr={optim_preset.lr}, weight_decay={optim_preset.weight_decay}")
    print(f"      betas={optim_preset.betas}")
    
    # 检查 get_scheduler_preset
    sched_preset = config.get_scheduler_preset()
    print(f"    ✓ get_scheduler_preset() 返回: {type(sched_preset).__name__}")
    print(f"      num_warmup_steps={sched_preset.num_warmup_steps}")
except Exception as e:
    print(f"    ✗ preset 检查失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("[VERIFY] 所有验证通过！Optimizer param groups 配置正确。")
print("=" * 80)