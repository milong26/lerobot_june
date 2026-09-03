"""
LLaVA-Pythia-400M Visual Embedding Extractor

Uses transformers to load lesjie/Llava-Pythia-400M and extract episode-level
visual embeddings from global (top) and wrist camera images.

Model loading:
- HuggingFace model: lesjie/Llava-Pythia-400M
- Loaded via AutoModelForImageTextToText + AutoProcessor
- Visual embedding extracted from the VLM's vision encoder output (last hidden state)
- Only visual features are used, no text generation involved
- Input: LeRobotDataset camera images (observation.images.top, observation.images.wrist)
- Output: unified JSON + PT format via BaseVLMExtractor.save_embedding()
"""

import sys
from pathlib import Path
from typing import Tuple

import torch
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

from differentvlm.vlm_extract.base_extractor import BaseVLMExtractor


class LlavaPythiaExtractor(BaseVLMExtractor):
    """LLaVA-Pythia-400M embedding extractor.

    Extracts visual embeddings using the LLaVA-Pythia-400M vision encoder.
    The embedding comes from the vision tower's last hidden state, NOT from text generation.
    """

    def __init__(self, model_id: str = "lesjie/Llava-Pythia-400M", **kwargs):
        super().__init__(model_id=model_id, **kwargs)

    def load_model(self) -> Tuple[torch.nn.Module, object]:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"[LLaVA-Pythia] Loading model: {self.model_id}")
        sys.stdout.flush()

        print(f"[LLaVA-Pythia] Loading processor...")
        sys.stdout.flush()
        processor = AutoProcessor.from_pretrained(self.model_id)
        print(f"[LLaVA-Pythia] Processor loaded")
        sys.stdout.flush()

        print(f"[LLaVA-Pythia] Loading model weights (float16, device={self.device})...")
        sys.stdout.flush()
        model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        print(f"[LLaVA-Pythia] Model loaded on {self.device}")
        sys.stdout.flush()

        self.model = model
        self.processor = processor
        return model, processor

    def encode_image(self, image: torch.Tensor) -> np.ndarray:
        """
        Encode a single image to visual embedding using LLaVA-Pythia-400M.

        Flow: image -> processor -> model (output_hidden_states=True) -> vision hidden state -> mean pool
        This extracts the VISION encoder output, NOT text embedding.
        """
        if image.dim() == 3 and image.shape[-1] in [1, 3, 4]:
            image = image.permute(2, 0, 1)

        image_cpu = image.cpu()

        inputs = self.processor(
            text="",
            images=image_cpu,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states

            last_hidden = hidden_states[-1]

            embedding = last_hidden.mean(dim=1).cpu().numpy().squeeze()

        return embedding