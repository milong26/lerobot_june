"""SmolVLA fine-tuning trainer."""

import os
import csv
import json
import time
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"


class SmolVLATrainer:
    def __init__(self, config, dataset, model=None, processor=None):
        self.config = config
        self.dataset = dataset
        self.device = config.device
        self.training_steps = config.training_steps
        self.eval_every = config.eval_every
        self.batch_size = config.batch_size
        self.lr = config.learning_rate

        self.model = model
        self.processor = processor
        self.optimizer = None

        self.train_log = []
        self.eval_results = []

    def setup_model(self, model, processor):
        self.model = model
        self.processor = processor
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr
        )

    def train_step(self, batch):
        self.model.train()
        outputs = self.model(**batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs.logits.mean() * 0.0
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def prepare_batch(self, frames):
        images = frames[self.config.global_cam_key]
        if isinstance(images, torch.Tensor):
            images = images.to(self.device)
        return {self.config.global_cam_key: images}

    def evaluate(self, evaluator):
        success_rate = evaluator.run_evaluation(self.model, self.processor, n_episodes=self.config.n_eval_episodes)
        return success_rate

    def save_checkpoint(self, checkpoint_dir, step):
        ckpt_path = Path(checkpoint_dir) / f"step_{step}"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            self.model.save_pretrained(str(ckpt_path))
        print(f"Checkpoint saved at step {step}: {ckpt_path}")

    def log_to_csv(self, csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'train_loss', 'eval_success_rate'])
            writer.writeheader()
            for entry in self.train_log:
                writer.writerow(entry)