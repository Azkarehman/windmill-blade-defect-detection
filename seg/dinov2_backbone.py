"""DINOv2 vision encoder as an mmseg backbone, with PEFT-LoRA fine-tuning.

Drop-in counterpart to Sam3Backbone — same 4 multi-scale features at 256 ch
each, same scales [288², 144², 72², 36²] for a 1008² input. DINOv2 has no
multi-scale neck of its own (it's a plain ViT), so we add a SimpleFPN-style
head (per ViTDet / Li et al. 2022) that takes the trunk's final patch grid at
72² and projects + up/down-samples to 4 scales.

Trainable params:
    - LoRA adapters on every block's `attention.attention.{query,key,value}`
      and `attention.output.dense` (r=8 default).
    - The SimpleFPN neck (~5M params) — small, fully trainable.
    - The decode head (added by mmseg config) — fully trainable.

Trunk's ~300M ViT params (dinov2-large) stay frozen.
"""
from typing import List

import torch
import torch.nn as nn

from mmengine.model import BaseModule
from mmseg.registry import MODELS


_DINOV2_HF = "facebook/dinov2-large"
_PATCH = 14
_TRUNK_DIM = 1024  # dinov2-large hidden size


def _inject_lora(trunk: nn.Module, r: int, alpha: int, dropout: float) -> nn.Module:
    """Wrap DINOv2 attention Linears with PEFT LoRA."""
    from peft import LoraConfig, get_peft_model

    # HF Dinov2Model layer names:
    #   encoder.layer.<i>.attention.attention.{query,key,value}
    #   encoder.layer.<i>.attention.output.dense
    target_modules: List[str] = []
    for name, mod in trunk.named_modules():
        if isinstance(mod, nn.Linear) and any(
            name.endswith(suf) for suf in (
                ".attention.attention.query",
                ".attention.attention.key",
                ".attention.attention.value",
                ".attention.output.dense",
            )
        ):
            target_modules.append(name)
    if not target_modules:
        raise RuntimeError("No LoRA target Linear layers found in DINOv2 trunk")

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type=None,
    )
    return get_peft_model(trunk, config)


class SimpleFPN(nn.Module):
    """ViTDet-style multi-scale neck.

    Takes a single ViT patch-grid feature map BCHW (e.g. 1024×72×72) and
    produces 4 outputs at 256 channels each, at scales 4× / 2× / 1× / 0.5×
    of the input grid.
    """

    def __init__(self, in_ch: int = _TRUNK_DIM, out_ch: int = 256):
        super().__init__()
        c2, c4 = in_ch // 2, in_ch // 4

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, c2, kernel_size=2, stride=2),
            nn.GroupNorm(32, c2),
            nn.GELU(),
            nn.ConvTranspose2d(c2, c4, kernel_size=2, stride=2),
        )
        self.up2 = nn.ConvTranspose2d(in_ch, c2, kernel_size=2, stride=2)
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Lateral 1×1 projections to the unified out_ch
        self.lat4 = nn.Conv2d(c4, out_ch, kernel_size=1)
        self.lat2 = nn.Conv2d(c2, out_ch, kernel_size=1)
        self.lat1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.lat05 = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [
            self.lat4(self.up4(x)),
            self.lat2(self.up2(x)),
            self.lat1(x),
            self.lat05(self.down2(x)),
        ]


@MODELS.register_module()
class Dinov2Backbone(BaseModule):
    """DINOv2 + SimpleFPN, mmseg-compatible.

    Args:
        hf_id: HuggingFace model id (default `facebook/dinov2-large`).
        lora_r, lora_alpha, lora_dropout: PEFT LoRA hyperparams. lora_r=0
            disables LoRA (trunk fully frozen, only neck + head train).
        freeze_trunk_other_than_lora: if True, trunk params (except LoRA
            adapters) get requires_grad=False.
        train_neck: if True, the SimpleFPN convs are trainable. Recommended True.
    """

    def __init__(
        self,
        hf_id: str = _DINOV2_HF,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        freeze_trunk_other_than_lora: bool = True,
        train_neck: bool = True,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        from transformers import AutoModel

        trunk = AutoModel.from_pretrained(hf_id)
        # We only need the patch encoder, not the pooled-output head.
        if getattr(trunk, "pooler", None) is not None:
            trunk.pooler = None

        if lora_r > 0:
            trunk = _inject_lora(trunk, lora_r, lora_alpha, lora_dropout)
        elif freeze_trunk_other_than_lora:
            for p in trunk.parameters():
                p.requires_grad = False

        self.trunk = trunk
        self.neck = SimpleFPN(in_ch=_TRUNK_DIM, out_ch=256)
        if not train_neck:
            for p in self.neck.parameters():
                p.requires_grad = False

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[Dinov2Backbone] total {total/1e6:.1f}M params  | "
            f"trainable {trainable/1e6:.1f}M  ({100*trainable/total:.2f}%)"
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """x: [B, 3, H, W] with H, W multiples of 14. Returns 4 feature maps."""
        B, _, H, W = x.shape
        if H % _PATCH or W % _PATCH:
            raise ValueError(f"DINOv2 expects H,W divisible by {_PATCH}; got {H}x{W}")
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=x.is_cuda
        ):
            out = self.trunk(pixel_values=x)
            tokens = out.last_hidden_state          # B, 1+Hp*Wp, C   (drops CLS)
            tokens = tokens[:, 1:, :]
            Hp, Wp = H // _PATCH, W // _PATCH
            C = tokens.shape[-1]
            feat = tokens.reshape(B, Hp, Wp, C).permute(0, 3, 1, 2).contiguous()
        feat = feat.float()
        return tuple(self.neck(feat))
