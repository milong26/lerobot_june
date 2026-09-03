"""
Prismatic-Qwen2.5-0.5B Visual Embedding Extractor

Loads Prismatic VLM with vision encoder + projector + Qwen2.5-0.5B backbone
and extracts episode-level visual embeddings from global and wrist camera images.

Architecture:
    image -> vision_encoder (SigLIP/CLIP) -> projector (MLP) -> visual_feature

Loading strategy (with fallback):
    1. Try HuggingFace AutoModelForImageTextToText (lucidrains/prismatic-qwen2.5-0.5b)
    2. Try prismatic library if available
    3. Manual construction: vision encoder + projector + Qwen LLM

Visual embedding extraction:
    image -> vision_encoder -> projector -> visual_feature
    Does NOT use Qwen language hidden states as visual embedding.

Input: LeRobotDataset camera images (observation.images.top, observation.images.wrist)
Output: unified JSON + PT format via BaseVLMExtractor.save_embedding()
"""

import sys
from pathlib import Path
from typing import Tuple

import torch
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

from differentvlm.vlm_extract.base_extractor import BaseVLMExtractor


class PrismaticQwenExtractor(BaseVLMExtractor):
    """Prismatic-Qwen2.5-0.5B embedding extractor.

    Extracts visual embeddings using the Prismatic VLM architecture:
    image -> vision_encoder -> projector -> visual_feature
    """

    def __init__(
        self,
        model_id: str = "lucidrains/prismatic-qwen2.5-0.5b",
        base_model_id: str = "Qwen/Qwen2.5-0.5B",
        vision_model_id: str = "google/siglip-so400m-patch14-384",
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.base_model_id = base_model_id
        self.vision_model_id = vision_model_id
        self.load_strategy = "unknown"

    def load_model(self) -> Tuple[torch.nn.Module, object]:
        print(f"[Prismatic-Qwen] Loading model: {self.model_id}")
        print(f"[Prismatic-Qwen] Architecture: vision_encoder -> projector -> Qwen LLM")
        sys.stdout.flush()

        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            print(f"[Prismatic-Qwen] Strategy 1: AutoModelForImageTextToText...")
            sys.stdout.flush()

            processor = AutoProcessor.from_pretrained(self.model_id)
            print(f"[Prismatic-Qwen]   Processor loaded")
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
            print(f"[Prismatic-Qwen]   Model loaded on {self.device} via AutoModelForImageTextToText")
            sys.stdout.flush()

            self.load_strategy = "AutoModelForImageTextToText"
            self.model = model
            self.processor = processor
            return model, processor

        except Exception as e:
            print(f"[Prismatic-Qwen]   Strategy 1 failed: {e}")
            sys.stdout.flush()

        try:
            from prismatic import load as prismatic_load

            print(f"[Prismatic-Qwen] Strategy 2: prismatic library...")
            sys.stdout.flush()

            model = prismatic_load(self.model_id, pretrained_base=True)
            model = model.to(self.device)
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            print(f"[Prismatic-Qwen]   Model loaded via prismatic library on {self.device}")
            sys.stdout.flush()

            self.load_strategy = "prismatic_library"
            self.model = model
            self.processor = None
            return model, None

        except Exception as e2:
            print(f"[Prismatic-Qwen]   Strategy 2 failed: {e2}")
            sys.stdout.flush()

        print(f"[Prismatic-Qwen] Strategy 3: Manual Prismatic VLM construction...")
        sys.stdout.flush()
        return self._load_manual_prismatic()

    def _load_manual_prismatic(self) -> Tuple[torch.nn.Module, object]:
        """
        Manually construct Prismatic VLM: vision encoder + projector + Qwen LLM.

        Components:
        1. Vision encoder: google/siglip-so400m-patch14-384 (fallback: openai/clip-vit-large-patch14)
        2. Projector: linear layer mapping vision_dim -> LLM_dim
        3. Language backbone: Qwen/Qwen2.5-0.5B
        """
        from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM

        print(f"[Prismatic-Qwen]   Loading vision encoder: {self.vision_model_id}")
        sys.stdout.flush()

        try:
            vision_encoder = AutoModel.from_pretrained(
                self.vision_model_id,
                torch_dtype=torch.float16,
            )
            print(f"[Prismatic-Qwen]     Loaded: {self.vision_model_id}")
            sys.stdout.flush()
        except Exception:
            fallback_vision = "openai/clip-vit-large-patch14"
            print(f"[Prismatic-Qwen]     {self.vision_model_id} unavailable, fallback to {fallback_vision}")
            sys.stdout.flush()
            vision_encoder = AutoModel.from_pretrained(
                fallback_vision,
                torch_dtype=torch.float16,
            )
            self.vision_model_id = fallback_vision

        vision_encoder = vision_encoder.to(self.device)
        for param in vision_encoder.parameters():
            param.requires_grad = False
        vision_encoder.eval()
        print(f"[Prismatic-Qwen]   Vision encoder loaded on {self.device}")
        sys.stdout.flush()

        processor = AutoProcessor.from_pretrained(self.vision_model_id)
        print(f"[Prismatic-Qwen]   Processor loaded")
        sys.stdout.flush()

        print(f"[Prismatic-Qwen]   Loading Qwen2.5-0.5B LLM: {self.base_model_id}")
        sys.stdout.flush()
        llm = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        for param in llm.parameters():
            param.requires_grad = False
        llm.eval()
        print(f"[Prismatic-Qwen]   LLM loaded on {self.device}")
        sys.stdout.flush()

        vision_dim = vision_encoder.config.hidden_size
        llm_dim = llm.config.hidden_size
        print(f"[Prismatic-Qwen]   Vision dim: {vision_dim}, LLM dim: {llm_dim}")
        sys.stdout.flush()

        projector = torch.nn.Linear(vision_dim, llm_dim)
        projector = projector.to(self.device).half()
        print(f"[Prismatic-Qwen]   Projector created: Linear({vision_dim} -> {llm_dim})")
        sys.stdout.flush()

        class ManualPrismaticVLM(torch.nn.Module):
            def __init__(self, vision_enc, proj, llm_model):
                super().__init__()
                self.vision_encoder = vision_enc
                self.projector = proj
                self.llm = llm_model

            def forward(self, **kwargs):
                return self.llm(**kwargs)

        model = ManualPrismaticVLM(vision_encoder, projector, llm)
        model.eval()

        self.model = model
        self.processor = processor
        self.load_strategy = "manual_construction"

        print(f"[Prismatic-Qwen] Manual Prismatic VLM constructed successfully")
        print(f"[Prismatic-Qwen]   Vision encoder: {self.vision_model_id}")
        print(f"[Prismatic-Qwen]   Projector: Linear({vision_dim} -> {llm_dim})")
        print(f"[Prismatic-Qwen]   LLM: {self.base_model_id}")
        sys.stdout.flush()
        return model, processor

    def encode_image(self, image: torch.Tensor) -> np.ndarray:
        """
        Encode a single image to visual embedding using Prismatic VLM.

        Flow: image -> vision_encoder -> projector -> visual_feature
        Does NOT use Qwen language hidden states.
        """
        if image.dim() == 3 and image.shape[-1] in [1, 3, 4]:
            image = image.permute(2, 0, 1)

        image_cpu = image.cpu()

        if self.load_strategy == "AutoModelForImageTextToText":
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

        elif self.load_strategy == "prismatic_library":
            with torch.no_grad():
                if hasattr(self.model, "vision_backbone"):
                    features = self.model.vision_backbone(image_cpu.unsqueeze(0).to(self.device))
                    if hasattr(features, "last_hidden_state"):
                        pooled = features.last_hidden_state.mean(dim=1)
                    elif isinstance(features, torch.Tensor):
                        pooled = features.mean(dim=1)
                    else:
                        pooled = features[0].mean(dim=1)
                else:
                    inputs = self.processor(images=image_cpu, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(self.device)
                    vision_out = self.model.vision_encoder(pixel_values=pixel_values)
                    if hasattr(vision_out, "last_hidden_state"):
                        pooled = vision_out.last_hidden_state.mean(dim=1)
                    else:
                        pooled = vision_out[0].mean(dim=1)

                if hasattr(self.model, "projector"):
                    projected = self.model.projector(pooled)
                else:
                    projected = pooled

                embedding = projected.cpu().numpy().squeeze()

        else:
            with torch.no_grad():
                inputs = self.processor(images=image_cpu, return_tensors="pt")
                if "pixel_values" in inputs:
                    pixel_values = inputs["pixel_values"].to(self.device)
                else:
                    pixel_values = image_cpu.unsqueeze(0).to(self.device)

                vision_outputs = self.model.vision_encoder(pixel_values=pixel_values)
                if hasattr(vision_outputs, "last_hidden_state"):
                    features = vision_outputs.last_hidden_state
                else:
                    features = vision_outputs[0]
                pooled = features.mean(dim=1)

                projected = self.model.projector(pooled)
                embedding = projected.cpu().numpy().squeeze()

        return embedding