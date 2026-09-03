"""
Prismatic-Qwen2.5-0.5B Visual Embedding Extractor

Loads Prismatic VLM with vision encoder + projector + Qwen2.5-0.5B backbone
and extracts episode-level visual embeddings from global and wrist camera images.
Uses the standard Prismatic VLM loading pattern.
"""

import sys
import os
from pathlib import Path
from typing import Tuple

import torch
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

from differentvlm.vlm_extract.base_extractor import BaseVLMExtractor


class PrismaticQwenExtractor(BaseVLMExtractor):
    """Prismatic-Qwen2.5-0.5B embedding extractor."""

    def __init__(
        self,
        model_id: str = "lucidrains/prismatic-qwen2.5-0.5b",
        base_model_id: str = "Qwen/Qwen2.5-0.5B",
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.base_model_id = base_model_id

    def load_model(self) -> Tuple[torch.nn.Module, object]:
        print(f"Loading Prismatic-Qwen2.5-0.5B: {self.model_id}")
        sys.stdout.flush()

        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            processor = AutoProcessor.from_pretrained(self.model_id)
            print("  Processor loaded")
            sys.stdout.flush()

            model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            )
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            print(f"  Model loaded on {self.device} via AutoModelForImageTextToText")
            sys.stdout.flush()

            self.model = model
            self.processor = processor
            return model, processor

        except Exception as e:
            print(f"  AutoModelForImageTextToText failed: {e}")
            print("  Trying prismatic library...")
            sys.stdout.flush()

        try:
            from prismatic import load as prismatic_load
            from prismatic.models import load as prismatic_load_v2

            model = prismatic_load(self.model_id, pretrained_base=True)
            model = model.to(self.device)
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            print(f"  Model loaded via prismatic library on {self.device}")
            sys.stdout.flush()

            self.model = model
            self.processor = None
            return model, None

        except Exception as e2:
            print(f"  Prismatic library also failed: {e2}")
            print("  Trying manual Prismatic VLM construction...")
            sys.stdout.flush()

        return self._load_manual_prismatic()

    def _load_manual_prismatic(self) -> Tuple[torch.nn.Module, object]:
        """
        Manually construct Prismatic VLM: vision encoder + projector + Qwen LLM.
        Prismatic uses SigLIP/CLIP vision encoder, MLP projector, and Qwen LLM.
        """
        from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM

        print("  Loading SigLIP vision encoder (openai/clip-vit-large-patch14)...")
        sys.stdout.flush()

        try:
            vision_encoder = AutoModel.from_pretrained(
                "google/siglip-so400m-patch14-384",
                torch_dtype=torch.float16,
            )
        except Exception:
            print("  SigLIP not available, trying CLIP...")
            sys.stdout.flush()
            vision_encoder = AutoModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=torch.float16,
            )

        vision_encoder = vision_encoder.to(self.device)
        for param in vision_encoder.parameters():
            param.requires_grad = False
        vision_encoder.eval()
        print("  Vision encoder loaded")
        sys.stdout.flush()

        processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        print("  Processor loaded")
        sys.stdout.flush()

        print(f"  Loading Qwen2.5-0.5B LLM: {self.base_model_id}")
        sys.stdout.flush()
        llm = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        for param in llm.parameters():
            param.requires_grad = False
        llm.eval()
        print("  LLM loaded")
        sys.stdout.flush()

        class ManualPrismaticVLM(torch.nn.Module):
            def __init__(self, vision_enc, proj, llm_model):
                super().__init__()
                self.vision_encoder = vision_enc
                self.projector = proj
                self.llm = llm_model

            def forward(self, **kwargs):
                return self.llm(**kwargs)

        projector = torch.nn.Linear(vision_encoder.config.hidden_size, llm.config.hidden_size)
        projector = projector.to(self.device).half()

        model = ManualPrismaticVLM(vision_encoder, projector, llm)
        model.eval()

        self.model = model
        self.processor = processor
        print("  Manual Prismatic VLM constructed")
        sys.stdout.flush()
        return model, processor

    def encode_image(self, image: torch.Tensor) -> np.ndarray:
        """Encode a single image to embedding using Prismatic VLM."""
        if image.dim() == 3 and image.shape[-1] in [1, 3, 4]:
            image = image.permute(2, 0, 1)

        if self.processor is not None:
            image_cpu = image.cpu()
            inputs = self.processor(
                images=image_cpu,
                return_tensors="pt",
            )
            if "pixel_values" in inputs:
                pixel_values = inputs["pixel_values"].to(self.device)
            else:
                pixel_values = image_cpu.unsqueeze(0).to(self.device)

            with torch.no_grad():
                if hasattr(self.model, "vision_encoder"):
                    vision_outputs = self.model.vision_encoder(pixel_values=pixel_values)
                    if hasattr(vision_outputs, "last_hidden_state"):
                        features = vision_outputs.last_hidden_state
                    else:
                        features = vision_outputs[0]
                    pooled = features.mean(dim=1)
                    if hasattr(self.model, "projector"):
                        projected = self.model.projector(pooled)
                    else:
                        projected = pooled
                    embedding = projected.cpu().numpy().squeeze()
                else:
                    inputs_full = self.processor(
                        text="Describe this image.",
                        images=image_cpu,
                        return_tensors="pt",
                    ).to(self.device)
                    outputs = self.model(**inputs_full, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]
                    embedding = hidden_states.mean(dim=1).cpu().numpy().squeeze()
        else:
            image_cpu = image.cpu()
            pixel_values = image_cpu.unsqueeze(0).to(self.device)

            with torch.no_grad():
                if hasattr(self.model, "vision_encoder"):
                    vision_outputs = self.model.vision_encoder(pixel_values=pixel_values)
                    if hasattr(vision_outputs, "last_hidden_state"):
                        features = vision_outputs.last_hidden_state
                    else:
                        features = vision_outputs[0]
                    pooled = features.mean(dim=1)
                    if hasattr(self.model, "projector"):
                        projected = self.model.projector(pooled)
                    else:
                        projected = pooled
                    embedding = projected.cpu().numpy().squeeze()
                else:
                    inputs_full = self.processor(
                        text="Describe this image.",
                        images=image_cpu,
                        return_tensors="pt",
                    ).to(self.device)
                    outputs = self.model(**inputs_full, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]
                    embedding = hidden_states.mean(dim=1).cpu().numpy().squeeze()

        return embedding