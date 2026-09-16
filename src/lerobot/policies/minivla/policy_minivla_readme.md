# MiniVLA Policy

This directory contains the LeRobot integration of MiniVLA.

## Scope

The implementation follows the official MiniVLA design:

- Vision backbone: DINO/SigLIP visual encoder
- Language backbone: Qwen2.5-0.5B based VLM
- Visual-language-action fusion
- Action tokenizer and VQ action support

## Configuration principles

The policy keeps model-specific configuration inside this directory. External training scripts should only provide dataset paths, devices, and runtime options.

Default settings:

- Base learning rate: `2e-5`
- Optimizer: AdamW
- Scheduler: constant-with-warmup style configuration
- Action normalization: quantile normalization
- Visual input: explicit camera key configuration

## Initialization modes

MiniVLA supports:

1. Scratch initialization
   - initialize policy parameters without official checkpoint weights.

2. Official backbone initialization
   - load compatible vision/projector/language backbone weights.
   - action tokenizer remains dataset compatible.

3. VQ action mode
   - use compatible VQ action tokenizer checkpoints.

## Design goal

This implementation provides a reusable LeRobot policy interface for different robot datasets. Dataset-specific assumptions should not be hard-coded in this directory.
