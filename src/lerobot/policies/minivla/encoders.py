"""
encoders.py

Official DINOv2 + SigLIP dual vision backbone for MiniVLA.
Mirrors teach_code/MiniVLA/prismatic/models/backbones/vision/dinosiglip_vit.py and base_vision.py.

Key design:
  - timm vit_large_patch14_reg4_dinov2.lvd142m + vit_so400m_patch14_siglip_224
  - get_intermediate_layers for second-to-last block output
  - resize-naive: direct resize to 224x224, SigLIP first Resize corrected
  - compute_sequence_patches for multi-image (T2, wrist)
  - concatenated DINO + SigLIP patch features on last dimension
  - get_image_transform() returns official DinoSigLIPImageTransform for processor use
  - build_dinosiglip_image_transform() pure factory (no model weights) for processor
"""

from __future__ import annotations

from functools import partial
from typing import Dict, Union

import timm
import torch
import torch.nn as nn
from PIL import Image
from timm.models.vision_transformer import VisionTransformer
from torchvision.transforms import Compose, Resize


# ---------------------------------------------------------------------------
# TIMM model identifiers (official MiniVLA)
# ---------------------------------------------------------------------------
DINOSIGLIP_TIMM_IDS = {
    "dinosiglip-vit-so-224px": {
        "dino": "vit_large_patch14_reg4_dinov2.lvd142m",
        "siglip": "vit_so400m_patch14_siglip_224",
    },
    "dinosiglip-vit-so-384px": {
        "dino": "vit_large_patch14_reg4_dinov2.lvd142m",
        "siglip": "vit_so400m_patch14_siglip_384",
    },
}


# ---------------------------------------------------------------------------
# Helper: unpack tuple from get_intermediate_layers
# ---------------------------------------------------------------------------
def _unpack_tuple(fn):
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result
    return wrapper


# ---------------------------------------------------------------------------
# Pure transform factory (no model weights loaded)
# ---------------------------------------------------------------------------
def build_dinosiglip_image_transform(
    vision_backbone_id: str = "dinosiglip-vit-so-224px",
    image_resize_strategy: str = "resize-naive",
    image_size: int = 224,
) -> "DinoSigLIPImageTransform":
    """
    Build DINO+SigLIP image transforms WITHOUT loading vision model weights.
    Mirrors teach_code/MiniVLA/prismatic/models/backbones/vision/dinosiglip_vit.py
    transform construction logic.

    Only uses TIMM pretrained data config to construct transforms.
    Does NOT instantiate DINOSigLIPViTBackbone or download model weights.
    """
    backbone_cfg = DINOSIGLIP_TIMM_IDS[vision_backbone_id]
    dino_timm_id = backbone_cfg["dino"]
    siglip_timm_id = backbone_cfg["siglip"]

    # Build dummy models just to resolve data config (timm needs the model to resolve config)
    # These are created with pretrained=False to avoid downloading weights
    dino_dummy = timm.create_model(dino_timm_id, pretrained=False, num_classes=0, img_size=image_size)
    siglip_dummy = timm.create_model(siglip_timm_id, pretrained=False, num_classes=0, img_size=image_size)

    dino_data_cfg = timm.data.resolve_model_data_config(dino_dummy)
    dino_data_cfg["input_size"] = (3, image_size, image_size)

    siglip_data_cfg = timm.data.resolve_model_data_config(siglip_dummy)
    siglip_data_cfg["input_size"] = (3, image_size, image_size)

    default_dino_transform = timm.data.create_transform(**dino_data_cfg, is_training=False)
    default_siglip_transform = timm.data.create_transform(**siglip_data_cfg, is_training=False)

    # Fix =>> SigLIP default transform resizes to *larger* than `image_size` (crops image)!
    assert isinstance(default_siglip_transform, Compose), "Unexpected `default_siglip_transform`!"
    assert isinstance(default_siglip_transform.transforms[0], Resize)
    default_siglip_transform = Compose(
        [
            Resize(image_size, interpolation=default_siglip_transform.transforms[0].interpolation),
            *default_siglip_transform.transforms[1:],
        ]
    )

    if image_resize_strategy == "resize-naive":
        assert isinstance(default_dino_transform, Compose), "Unexpected `default_dino_transform`!"
        assert isinstance(default_siglip_transform, Compose), "Unexpected `default_siglip_transform`!"
        assert isinstance(default_dino_transform.transforms[0], Resize)
        assert isinstance(default_siglip_transform.transforms[0], Resize)

        target_size = (image_size, image_size)
        dino_transform = Compose(
            [
                Resize(target_size, interpolation=default_dino_transform.transforms[0].interpolation),
                *default_dino_transform.transforms[1:],
            ]
        )
        siglip_transform = Compose(
            [
                Resize(target_size, interpolation=default_siglip_transform.transforms[0].interpolation),
                *default_siglip_transform.transforms[1:],
            ]
        )
        return DinoSigLIPImageTransform(dino_transform, siglip_transform)
    else:
        raise ValueError(f"image_resize_strategy '{image_resize_strategy}' is not supported!")


# ---------------------------------------------------------------------------
# Multi-image patch computation (official compute_sequence_patches)
# ---------------------------------------------------------------------------
def _merge_two_dims(tensor: torch.Tensor, start_dim: int = 1) -> torch.Tensor:
    """Merge two consecutive dimensions, e.g. (B, T, C, H, W) -> (B*T, C, H, W)."""
    shape = tensor.shape
    new_shape = shape[:start_dim] + (shape[start_dim] * shape[start_dim + 1],) + shape[start_dim + 2:]
    return tensor.reshape(new_shape)


def compute_sequence_patches(
    pixel_values: Dict[str, torch.Tensor],
    featurizers: Dict[str, nn.Module],
    seq_len: int,
) -> Dict[str, torch.Tensor]:
    """
    Official compute_sequence_patches semantics.
    pixel_values[k] shape: (B, T, C, H, W) with T >= seq_len.
    Returns patches[k] shape: (B, T*num_patches, embed_dim).
    """
    patches = {}
    for k in pixel_values:
        assert len(pixel_values[k].shape) == 5, f"Pixel values must be (B, T, C, H, W), got {pixel_values[k].shape}"
        assert pixel_values[k].shape[1] >= seq_len, f"Sequence length too short: {pixel_values[k].shape[1]} < {seq_len}"
        trunc = pixel_values[k][:, :seq_len]  # (B, T, C, H, W)
        # (B*T, C, H, W) -> (B*T, num_patches, embed_dim)
        bt = _merge_two_dims(trunc, start_dim=0)
        out = featurizers[k](bt)
        # (B*T, num_patches, embed_dim) -> (B, T*num_patches, embed_dim)
        b, t = trunc.shape[0], trunc.shape[1]
        n_tokens = out.shape[1]
        out = out.view(b, t * n_tokens, -1)
        patches[k] = out
    return patches


# ---------------------------------------------------------------------------
# Image transform dataclass (mirrors DinoSigLIPImageTransform from official code)
# ---------------------------------------------------------------------------
class DinoSigLIPImageTransform:
    """
    Holds both DINO and SigLIP transforms; returns a dict.
    Mirrors teach_code/MiniVLA/prismatic/models/backbones/vision/dinosiglip_vit.py::DinoSigLIPImageTransform.
    Accepts PIL Image or torch.Tensor input.
    """

    def __init__(self, dino_transform: Compose, siglip_transform: Compose):
        self.dino_transform = dino_transform
        self.siglip_transform = siglip_transform

    def __call__(
        self, img: Union[Image.Image, torch.Tensor], **kwargs: str
    ) -> Dict[str, torch.Tensor]:
        return {
            "dino": self.dino_transform(img),
            "siglip": self.siglip_transform(img),
        }

    def apply_pil_image(self, img: Image.Image) -> Dict[str, torch.Tensor]:
        """Apply transforms to a PIL Image input."""
        return {
            "dino": self.dino_transform(img),
            "siglip": self.siglip_transform(img),
        }

    def apply_tensor_image(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply transforms to a torch.Tensor input [C, H, W] in float [0,1].
        The official TIMM Compose includes ToTensor which is a no-op for tensors.
        We manually execute the same operations: Resize -> (ToTensor no-op) -> Normalize.
        """
        results = {}
        for branch, transform in [("dino", self.dino_transform), ("siglip", self.siglip_transform)]:
            out = img
            for t in transform.transforms:
                if isinstance(t, Resize):
                    out = t(out)
                elif isinstance(t, nn.Module) and hasattr(t, "forward"):
                    out = t(out)
                elif hasattr(t, "__call__"):
                    try:
                        out = t(out)
                    except Exception:
                        pass
            results[branch] = out
        return results


# ---------------------------------------------------------------------------
# DINO-SigLIP ViT Backbone
# ---------------------------------------------------------------------------
class DINOSigLIPViTBackbone(nn.Module):
    """
    Official DINO-SigLIP dual ViT backbone.
    Returns concatenated patch features: torch.cat([dino_patches, siglip_patches], dim=-1).
    Mirrors teach_code/MiniVLA/prismatic/models/backbones/vision/dinosiglip_vit.py::DinoSigLIPViTBackbone.
    """

    def __init__(
        self,
        vision_backbone_id: str = "dinosiglip-vit-so-224px",
        image_resize_strategy: str = "resize-naive",
        default_image_size: int = 224,
        image_sequence_len: int = 1,
    ):
        super().__init__()
        self.vision_backbone_id = vision_backbone_id
        self.image_resize_strategy = image_resize_strategy
        self.default_image_size = default_image_size
        self.image_sequence_len = image_sequence_len

        backbone_cfg = DINOSIGLIP_TIMM_IDS[vision_backbone_id]
        dino_timm_id = backbone_cfg["dino"]
        siglip_timm_id = backbone_cfg["siglip"]

        # --- DINOv2 featurizer ---
        self.dino_featurizer: VisionTransformer = timm.create_model(
            dino_timm_id, pretrained=True, num_classes=0, img_size=default_image_size
        )
        self.dino_featurizer.eval()
        # Monkey-patch forward to use second-to-last block
        self.dino_featurizer.forward = _unpack_tuple(
            partial(
                self.dino_featurizer.get_intermediate_layers,
                n={len(self.dino_featurizer.blocks) - 2},
            )
        )

        # --- SigLIP featurizer ---
        self.siglip_featurizer: VisionTransformer = timm.create_model(
            siglip_timm_id, pretrained=True, num_classes=0, img_size=default_image_size
        )
        self.siglip_featurizer.eval()
        self.siglip_featurizer.forward = _unpack_tuple(
            partial(
                self.siglip_featurizer.get_intermediate_layers,
                n={len(self.siglip_featurizer.blocks) - 2},
            )
        )

        # --- Data configs ---
        self.dino_data_cfg = timm.data.resolve_model_data_config(self.dino_featurizer)
        self.dino_data_cfg["input_size"] = (3, default_image_size, default_image_size)

        self.siglip_data_cfg = timm.data.resolve_model_data_config(self.siglip_featurizer)
        self.siglip_data_cfg["input_size"] = (3, default_image_size, default_image_size)

        # --- Default transforms ---
        default_dino_transform = timm.data.create_transform(**self.dino_data_cfg, is_training=False)
        default_siglip_transform = timm.data.create_transform(**self.siglip_data_cfg, is_training=False)

        # Fix =>> SigLIP default transform resizes to *larger* than `self.default_image_size` (crops image)!
        # Official fix from teach_code/MiniVLA/prismatic/models/backbones/vision/dinosiglip_vit.py
        assert isinstance(default_siglip_transform, Compose), "Unexpected `default_siglip_transform`!"
        assert isinstance(default_siglip_transform.transforms[0], Resize)
        default_siglip_transform = Compose(
            [
                Resize(default_image_size, interpolation=default_siglip_transform.transforms[0].interpolation),
                *default_siglip_transform.transforms[1:],
            ]
        )

        # --- Apply resize-naive strategy ---
        if image_resize_strategy == "resize-naive":
            assert isinstance(default_dino_transform, Compose), "Unexpected `default_dino_transform`!"
            assert isinstance(default_siglip_transform, Compose), "Unexpected `default_siglip_transform`!"
            assert isinstance(default_dino_transform.transforms[0], Resize)
            assert isinstance(default_siglip_transform.transforms[0], Resize)

            target_size = (default_image_size, default_image_size)
            dino_transform = Compose(
                [
                    Resize(target_size, interpolation=default_dino_transform.transforms[0].interpolation),
                    *default_dino_transform.transforms[1:],
                ]
            )
            siglip_transform = Compose(
                [
                    Resize(target_size, interpolation=default_siglip_transform.transforms[0].interpolation),
                    *default_siglip_transform.transforms[1:],
                ]
            )
            self.image_transform = DinoSigLIPImageTransform(dino_transform, siglip_transform)
        else:
            raise ValueError(f"image_resize_strategy '{image_resize_strategy}' is not supported!")

    @property
    def default_image_resolution(self) -> tuple:
        return self.dino_data_cfg["input_size"]

    @property
    def embed_dim(self) -> int:
        return self.dino_featurizer.embed_dim + self.siglip_featurizer.embed_dim

    @property
    def num_patches(self) -> int:
        return self.dino_featurizer.patch_embed.num_patches * self.image_sequence_len

    def get_image_transform(self) -> DinoSigLIPImageTransform:
        """
        Standard get_image_transform() interface.
        Returns the official DinoSigLIPImageTransform for use by processor or direct inference.
        Mirrors teach_code/MiniVLA/prismatic/models/backbones/vision/base_vision.py::VisionBackbone.get_image_transform.
        """
        return self.image_transform

    def forward(self, pixel_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        pixel_values: {"dino": tensor, "siglip": tensor}
          Single image: (B, C, H, W)
          Multi-image:  (B, T, C, H, W)
        Returns: (B, num_patches, dino_dim + siglip_dim)
        """
        dino_input = pixel_values["dino"]
        siglip_input = pixel_values["siglip"]

        # Shape validation
        if self.image_sequence_len == 1:
            assert dino_input.dim() == 4, f"DINO input must be (B, C, H, W), got {dino_input.shape}"
            assert siglip_input.dim() == 4, f"SigLIP input must be (B, C, H, W), got {siglip_input.shape}"
        else:
            assert dino_input.dim() == 5, f"DINO input must be (B, T, C, H, W), got {dino_input.shape}"
            assert siglip_input.dim() == 5, f"SigLIP input must be (B, T, C, H, W), got {siglip_input.shape}"
            assert dino_input.shape[1] >= self.image_sequence_len, (
                f"DINO sequence length {dino_input.shape[1]} < required {self.image_sequence_len}"
            )
            assert siglip_input.shape[1] >= self.image_sequence_len, (
                f"SigLIP sequence length {siglip_input.shape[1]} < required {self.image_sequence_len}"
            )

        # DINO/SigLIP patch count consistency check
        dino_num_patches = self.dino_featurizer.patch_embed.num_patches
        siglip_num_patches = self.siglip_featurizer.patch_embed.num_patches
        assert dino_num_patches == siglip_num_patches, (
            f"DINO ({dino_num_patches}) and SigLIP ({siglip_num_patches}) patch counts must match!"
        )

        if self.image_sequence_len == 1:
            dino_patches = self.dino_featurizer(dino_input)
            siglip_patches = self.siglip_featurizer(siglip_input)
        else:
            featurizers = {"dino": self.dino_featurizer, "siglip": self.siglip_featurizer}
            patches = compute_sequence_patches(pixel_values, featurizers, self.image_sequence_len)
            dino_patches = patches["dino"]
            siglip_patches = patches["siglip"]
        return torch.cat([dino_patches, siglip_patches], dim=-1)