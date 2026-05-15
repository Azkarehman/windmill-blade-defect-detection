# DMDD — SAM 3.1 LoRA for wind-blade defect segmentation

Fine-tunes Meta's **SAM 3.1** (848M params, 11M-image + 1B-mask pretrain) as a
semantic segmenter for wind-turbine blade defects, using a LoRA adapter on the
attention layers so only 2% of params are trainable. Drop-in replacement for the
team's SegFormer-B4 baseline within the same `mmsegmentation` data pipeline.

## What's here

| path | purpose |
|---|---|
| `seg/sam3_backbone.py` | mmseg backbone wrapping `Sam3DualViTDetNeck` → 4 multi-scale features at 256ch |
| `seg/_torch_compat.py` | monkey-patch for SAM 3.1's inference-only `addmm_act` so it works under autograd |
| `seg/windblade_merged.py` | mmseg dataset with 9→5 LUT merging + per-WTG leak filter |
| `seg/fast_val_hook_v2.py` | val-during-train hook (image-level recall, no OOM) |
| `seg/configs/sam3_lora_windblade_8class.py` | **headline experiment** — SAM 3.1 + LoRA, 8-class, 30k iters |
| `seg/configs/deeplabv3plus_r152_windblade_5class.py` | CNN comparison — ResNet-152 + DeepLabV3+, 5-class merged |
| `setup_on_new_server.sh` | env bootstrap (conda + torch 2.7 + mmseg + SAM 3.1 + LoRA peft) |
| `requirements.txt` | top-level pip deps |
| `EXPERIMENTS.md` | hypotheses, configs, results |
| `FAST_VAL_HOOK_V2.md` | design notes for the custom val hook |

## Approach

```
SAM 3.1 trunk (frozen, 446M)
       │
       └── LoRA r=8 α=16 on every attn.qkv + attn.proj  (9.4M trainable)
              │
              ▼
       Sam3DualViTDetNeck (trainable)
              │
              ▼
       SegformerHead (trainable, in_channels=[256,256,256,256])
              │
              ▼
       8-class logits (1× BG + 7 defect classes — team's taxonomy)
```

- Crop size **1008²** (SAM RoPE is fixed there).
- Optimizer / schedule / loss **identical to team's SegFormer config** (AdamW 6e-5, decode-head 10×, LinearLR→PolyLR, CE loss).
- Sampling **identical**: `ConcatDataset(full + RepeatDataset×5(rare))`, `BladeFilteredRandomCrop`, `RareClassCrop`, `RandomFlip`, `PhotoMetricDistortion`.
- 30k iters vs team's 300k — foundation-model fine-tune needs less.

