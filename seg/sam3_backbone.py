"""SAM 3.1 vision encoder as an mmseg backbone, with PEFT-LoRA fine-tuning.

What this is:
    Drop-in replacement for `mit_b4` in the team's SegFormer config. The trunk
    runs at SAM 3.1's native 1008×1008 resolution; the SimpleFPN neck produces
    4 feature maps at 256 channels each (scales 4×, 2×, 1×, 0.5× of the trunk's
    14-stride patch grid → spatial sizes [288, 144, 72, 36]).

Trainable params:
    - LoRA adapters on every block's `attn.qkv` and `attn.proj` (r=8 default).
    - The neck's conv stack (~7M params) — small, fully trainable.
    - The decode head (SegFormerHead, added by mmseg config) — fully trainable.

Trunk's 446M ViT params stay frozen.

Usage (in an mmseg config):

    custom_imports = dict(
        imports=['sam3_backbone'],
        allow_failed_imports=False,
    )

    model = dict(
        type='EncoderDecoder',
        backbone=dict(
            type='Sam3Backbone',
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            freeze_trunk_other_than_lora=True,
            train_neck=True,
        ),
        decode_head=dict(
            type='SegformerHead',
            in_channels=[256, 256, 256, 256],
            in_index=[0, 1, 2, 3],
            channels=256,
            num_classes=9,
            ...
        ),
    )
"""
import os
import warnings
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from mmengine.model import BaseModule
from mmseg.registry import MODELS


def _patch_sam3_addmm_act_for_training() -> None:
    """SAM 3.1's perflib has a fast-path that asserts grad is disabled (inference
    only). Replace it with a version that falls back to standard Linear+act
    when grad is enabled, so the MLP blocks work under autograd.
    """
    import sam3.perflib.fused as fused

    if getattr(fused, "_sam3_backbone_patched", False):
        return

    _orig = fused.addmm_act

    def _safe_addmm_act(activation, linear, mat1):
        if not torch.is_grad_enabled():
            return _orig(activation, linear, mat1)
        # Training fallback: standard fused Linear + activation
        x = linear(mat1)
        if activation in (torch.nn.functional.relu, torch.nn.ReLU):
            return torch.nn.functional.relu(x)
        if activation in (torch.nn.functional.gelu, torch.nn.GELU):
            return torch.nn.functional.gelu(x)
        raise ValueError(f"Unexpected activation {activation}")

    fused.addmm_act = _safe_addmm_act
    # Also rebind any modules that already imported the symbol directly.
    import sam3.model.vitdet as vitdet
    if hasattr(vitdet, "addmm_act"):
        vitdet.addmm_act = _safe_addmm_act
    fused._sam3_backbone_patched = True


def _build_sam3_vision_backbone() -> nn.Module:
    """Load SAM 3.1, return its (vision_backbone) Sam3DualViTDetNeck only.

    This is a heavy call (loads 848M-param model from HF). It runs once at
    backbone __init__; the parent text/decoder modules are GC'd.
    """
    import sam3
    from sam3 import build_sam3_image_model

    sam3_pkg = Path(sam3.__file__).resolve().parent
    candidates = [
        sam3_pkg / "assets" / "bpe_simple_vocab_16e6.txt.gz",
        sam3_pkg.parent / "assets" / "bpe_simple_vocab_16e6.txt.gz",
    ]
    bpe_path = next((p for p in candidates if p.exists()), None)
    if bpe_path is None:
        raise FileNotFoundError(f"BPE vocab not found in {candidates}")
    full = build_sam3_image_model(bpe_path=str(bpe_path))
    vb = full.backbone.vision_backbone  # Sam3DualViTDetNeck
    # Free the rest of the SAM 3.1 model — decoder, text encoder, etc.
    del full
    torch.cuda.empty_cache()
    return vb


def _inject_lora(trunk: nn.Module, r: int, alpha: int, dropout: float) -> nn.Module:
    """Wrap target Linear layers in `trunk` with PEFT LoRA. Returns trunk-as-PeftModel."""
    from peft import LoraConfig, get_peft_model

    # Target every attention block's qkv + proj. These names come from
    # sam3/model/vitdet.py: ViT.blocks[i].attn.qkv / attn.proj
    target_modules = []
    for name, mod in trunk.named_modules():
        if isinstance(mod, nn.Linear) and (
            name.endswith(".attn.qkv") or name.endswith(".attn.proj")
        ):
            target_modules.append(name)
    if not target_modules:
        raise RuntimeError("No LoRA target Linear layers found in trunk")

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type=None,  # generic — we're not using one of PEFT's preset tasks
    )
    return get_peft_model(trunk, config)


@MODELS.register_module()
class Sam3Backbone(BaseModule):
    """SAM 3.1 vision encoder wrapped for mmseg.

    Args:
        lora_r, lora_alpha, lora_dropout: PEFT LoRA hyperparams. Set lora_r=0
            to disable LoRA (encoder fully frozen, only neck + head train).
        freeze_trunk_other_than_lora: if True, trunk params (except LoRA
            adapters) get requires_grad=False.
        train_neck: if True, the SimpleFPN neck convs (~7M params) are
            trainable. Recommended True.
        init_cfg: mmseg init config (ignored — we load from HF).
    """

    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        freeze_trunk_other_than_lora: bool = True,
        train_neck: bool = True,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        _patch_sam3_addmm_act_for_training()
        vb = _build_sam3_vision_backbone()  # Sam3DualViTDetNeck

        # Convert trunk → PEFT model with LoRA adapters
        if lora_r > 0:
            vb.trunk = _inject_lora(vb.trunk, lora_r, lora_alpha, lora_dropout)
            # PEFT's get_peft_model freezes the base trunk and only enables
            # LoRA adapter params. Verify by listing trainable.
        elif freeze_trunk_other_than_lora:
            for p in vb.trunk.parameters():
                p.requires_grad = False

        # Optionally toggle neck (the SimpleFPN convs)
        if not train_neck:
            for p in vb.convs.parameters():
                p.requires_grad = False
            if vb.sam2_convs is not None:
                for p in vb.sam2_convs.parameters():
                    p.requires_grad = False

        self.vision_backbone = vb

        # Tally trainable params
        total = sum(p.numel() for p in vb.parameters())
        trainable = sum(p.numel() for p in vb.parameters() if p.requires_grad)
        print(
            f"[Sam3Backbone] total {total/1e6:.1f}M params  | "
            f"trainable {trainable/1e6:.1f}M  ({100*trainable/total:.2f}%)"
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """x: [B, 3, 1008, 1008]. Returns list of 4 feature maps."""
        # SAM 3.1 was trained with bf16 autocast active. mmseg's outer loop
        # may or may not have autocast on — we enable bf16 here defensively
        # so frozen weights don't get f32-promoted.
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=x.is_cuda,
        ):
            sam3_out, _, _, _ = self.vision_backbone.forward(x)
        # mmseg decoder heads typically expect float32 — cast back.
        return tuple(f.float() for f in sam3_out)
