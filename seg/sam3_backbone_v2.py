"""SAM 3.1 vision encoder as an mmseg backbone — v2 with last-N-block unfreeze.

Adds `unfreeze_last_n_blocks` on top of `Sam3Backbone`. After PEFT-LoRA injection
and the freeze step, the last N transformer blocks' parameters are switched back
to requires_grad=True (attn + MLP + norms). LoRA adapters on those blocks stay
trainable as before.

Registered as `Sam3BackboneV2` to avoid colliding with the v1 backbone in
mmseg's MODELS registry — use both in different configs side-by-side.

SAM 3.1 trunk depth = 32 blocks (embed_dim=1024, mlp_ratio=4.625), so each
block adds ~13.9M trainable params. With unfreeze_last_n_blocks=2 + LoRA + neck
the trainable count goes from v1's 9.4M to ~37M.
"""
import os
import warnings
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from mmengine.model import BaseModule
from mmseg.registry import MODELS

from sam3_backbone import (
    _patch_sam3_addmm_act_for_training,
    _build_sam3_vision_backbone,
    _inject_lora,
)


@MODELS.register_module()
class Sam3BackboneV2(BaseModule):
    """SAM 3.1 vision encoder, with optional unfreeze of last N transformer blocks.

    Args:
        lora_r, lora_alpha, lora_dropout: PEFT LoRA hyperparams. Set lora_r=0
            to disable LoRA.
        freeze_trunk_other_than_lora: if True, all trunk params (except LoRA
            adapters) get requires_grad=False.
        train_neck: if True, the SimpleFPN neck convs (~7M params) train.
        unfreeze_last_n_blocks: number of trailing transformer blocks to fully
            unfreeze (attn + MLP + norms). 0 = keep frozen (matches v1).
        init_cfg: mmseg init config (ignored — weights come from HF).
    """

    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        freeze_trunk_other_than_lora: bool = True,
        train_neck: bool = True,
        unfreeze_last_n_blocks: int = 0,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        _patch_sam3_addmm_act_for_training()
        vb = _build_sam3_vision_backbone()

        if lora_r > 0:
            vb.trunk = _inject_lora(vb.trunk, lora_r, lora_alpha, lora_dropout)
        elif freeze_trunk_other_than_lora:
            for p in vb.trunk.parameters():
                p.requires_grad = False

        if not train_neck:
            for p in vb.convs.parameters():
                p.requires_grad = False
            if vb.sam2_convs is not None:
                for p in vb.sam2_convs.parameters():
                    p.requires_grad = False

        if unfreeze_last_n_blocks > 0:
            # PEFT wraps the ViT: original blocks live at .base_model.model.blocks.
            # Without LoRA, blocks are directly under vb.trunk.
            base = vb.trunk.base_model.model if lora_r > 0 else vb.trunk
            blocks = base.blocks
            n_total = len(blocks)
            n = min(unfreeze_last_n_blocks, n_total)
            start = n_total - n
            for i in range(start, n_total):
                for p in blocks[i].parameters():
                    p.requires_grad = True
            print(f"[Sam3BackboneV2] unfroze blocks [{start}, {n_total}) "
                  f"out of {n_total}")

        self.vision_backbone = vb

        total = sum(p.numel() for p in vb.parameters())
        trainable = sum(p.numel() for p in vb.parameters() if p.requires_grad)
        print(
            f"[Sam3BackboneV2] total {total/1e6:.1f}M params  | "
            f"trainable {trainable/1e6:.1f}M  ({100*trainable/total:.2f}%)"
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=x.is_cuda,
        ):
            sam3_out, _, _, _ = self.vision_backbone.forward(x)
        return tuple(f.float() for f in sam3_out)
