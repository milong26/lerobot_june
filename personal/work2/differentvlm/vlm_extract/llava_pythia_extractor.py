"""
LLaVA-Pythia-400M Visual Embedding Extractor

Uses transformers to load lesjie/Llava-Pythia-400M and extract episode-level
visual embeddings from global (top) and wrist camera images.
"""

import sys
import os
from pathlib import Path
from typing import Tuple

import torch
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

from differentvlm.vlm_extract.base_extractor import BaseVLMExtractor


class LlavaPythiaExtractor(BaseVLMExtractor):
    """LLaVA-Pythia-400M embedding extractor."""

    def __init__(self, model_id: str = "lesjie/Llava-Pythia-400M", **kwargs):
        super().__init__(model_id=model_id, **kwargs)

    def load_model(self) -> Tuple[torch.nn.Module, object]:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"Loading LLaVA-Pythia-400M: {self.model_id}")
        sys.stdout.flush()

        processor = AutoProcessor.from_pretrained(self.model_id)
        print("  Processor loaded")
        sys.stdout.flush()

        model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        print(f"  Model loaded on {self.device}")
        sys.stdout.flush()

        self.model = model
        self.processor = processor
        return model, processor

    def encode_image(self, image: torch.Tensor) -> np.ndarray:
        """Encode a single image to embedding using LLaVA-Pythia-400M."""
        if image.dim() == 3 and image.shape[-1] in [1, 3, 4]:
            image = image.permute(2, 0, 1)

        image_cpu = image.cpu()

        inputs = self.processor(
            text="Describe this image.",
            images=image_cpu,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            embedding = hidden_states.mean(dim=1).cpu().numpy().squeeze()

        return embedding