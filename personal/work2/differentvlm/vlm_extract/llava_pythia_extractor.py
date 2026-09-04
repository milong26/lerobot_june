"""
LLaVA-Pythia-400M Visual Embedding Extractor

Uses lerobot's internal TinyVLA model classes to load lesjie/Llava-Pythia-400M
and extract episode-level visual embeddings from global (top) and wrist camera images.

Model loading:
- HuggingFace model: lesjie/Llava-Pythia-400M
- Loaded via lerobot.policies.tinyvla.llava_pythia LlavaPythiaForCausalLM
- Visual embedding extracted from: image -> vision_tower -> mm_projector -> visual feature
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

    Extracts visual embeddings using the LLaVA-Pythia-400M vision encoder + projector.
    Flow: image -> vision_tower -> mm_projector -> visual feature
    """

    def __init__(self, model_id: str = "lesjie/Llava-Pythia-400M", **kwargs):
        super().__init__(model_id=model_id, **kwargs)

    def load_model(self) -> Tuple[torch.nn.Module, object]:
        """
        Load LLaVA-Pythia model using lerobot's internal classes.

        The model uses a custom 'llava_pythia' architecture that transformers
        doesn't recognize natively. We use lerobot's internal model classes
        with trust_remote_code=True to load it properly.

        Components loaded:
        1. Vision tower: SiglipVisionTower or CLIPVisionTower
        2. Projector: mm_projector (MLP mapping vision_dim -> LLM_dim)
        3. Language backbone: GPTNeoX (Pythia)
        """
        from lerobot.policies.tinyvla.llava_pythia.model.language_model.pythia.llava_pythia import (
            LlavaPythiaConfig,
            LlavaPythiaForCausalLM,
        )
        import transformers

        print(f"[LLaVA-Pythia] Loading model: {self.model_id}")
        sys.stdout.flush()

        print(f"[LLaVA-Pythia] Loading config with trust_remote_code=True...")
        sys.stdout.flush()
        llava_config = LlavaPythiaConfig.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        print(f"[LLaVA-Pythia] Config loaded")
        sys.stdout.flush()

        print(f"[LLaVA-Pythia] Loading model weights (device={self.device})...")
        sys.stdout.flush()
        model = LlavaPythiaForCausalLM.from_pretrained(
            self.model_id,
            config=llava_config,
            trust_remote_code=True,
            _fast_init=False,
        )
        model.config.use_cache = False

        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        model = model.to(self.device)
        model = model.to(dtype=torch.float16)

        print(f"[LLaVA-Pythia] Model loaded on {self.device}")
        print(f"[LLaVA-Pythia] Vision tower: {type(model.get_model().vision_tower).__name__}")
        print(f"[LLaVA-Pythia] Projector: {model.get_model().mm_projector}")
        sys.stdout.flush()

        self.model = model
        self.processor = None

        print(f"[LLaVA-Pythia] Loading image processor...")
        sys.stdout.flush()
        self.image_processor = transformers.AutoImageProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        print(f"[LLaVA-Pythia] Image processor loaded")
        sys.stdout.flush()

        print(f"[LLaVA-Pythia] Compiling vision encoder + projector with torch.compile...")
        sys.stdout.flush()
        self.model.encode_images = torch.compile(
            self.model.encode_images,
            mode="reduce-overhead",
            fullgraph=False,
        )
        print(f"[LLaVA-Pythia] Model compiled for GPU acceleration")
        sys.stdout.flush()

        return model, self.image_processor

    def encode_image(self, image: torch.Tensor) -> np.ndarray:
        """
        Encode a single image to visual embedding using LLaVA-Pythia-400M.

        Flow: image -> vision_tower -> mm_projector -> visual feature -> mean pool
        This extracts the VISION encoder + projector output, NOT text embedding.
        """
        if image.dim() == 3 and image.shape[-1] in [1, 3, 4]:
            image = image.permute(2, 0, 1)

        # Force float32 for image processor, then convert to model dtype (float16) for GPU
        image_cpu = image.cpu().float()

        processed = self.image_processor(
            images=image_cpu,
            return_tensors="pt",
        )
        pixel_values = processed["pixel_values"].to(self.device, dtype=torch.float16)

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                image_features = self.model.encode_images(pixel_values, proj=True)

            if image_features.dim() == 3:
                embedding = image_features.mean(dim=1).cpu().numpy().squeeze()
            elif image_features.dim() == 2:
                embedding = image_features.cpu().numpy().squeeze()
            else:
                embedding = image_features.cpu().numpy().squeeze()

        return embedding