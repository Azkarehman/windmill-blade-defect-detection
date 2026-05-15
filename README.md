# Wind-blade defect detection — experiments

## Project context

- **Goal**: wind-turbine blade defect semantic segmentation. Customer team's
  production baseline = SegFormer-B4 (9-class). Customer target = **micro
  image-level recall ≥ 85%**.
- **Approach in this repo**: swap the SegFormer backbone for stronger pretrained
  vision encoders, fine-tune via LoRA on attention layers, keep everything else
  identical to the team's recipe (data pipeline, transforms, optimizer, schedule,
  loss). Three backbones tried:
    1. **SAM 3.1** (Meta, March 2026; 848 M params, 11 M-image + 1 B-mask pretrain)
    2. **DINOv2-Large** (Meta, 2023; 300 M, LVD-142M self-supervised)
    3. **ResNet-152 + DeepLabV3+** (ImageNet) — CNN baseline (no LoRA, full FT)
- **Data**: 9,564 train / 2,549 train_rare (×5 repeat) / 2,124 val
  (`val_diuid_7.json`, 7 WTGs) / 3,083 test (`test_diuid_10.json`, 10 WTGs).
  All splits WTG-disjoint (per-WTG leak fixed in `windblade_merged.py`).
- **Hardware**: 1× A100 80 GB PCIe.

---

## Exp 1 — SAM 3.1 zero-shot as segmenter (text prompts)

**Hypothesis**: SAM 3.1's text-prompted PCS mode can do defect segmentation
zero-shot, no fine-tune needed.

**Setup**: one text prompt per class
(`"wind turbine blade surface damage"`, `"crack on wind turbine blade"`, etc.),
`Sam3Processor`, score threshold 0.5 then 0.05 diagnostic. Defect-heavy test
images.

**Takeaway**: SAM 3.1 does not understand wind-turbine defect concepts
zero-shot with these prompts. Fine-tune required, which motivated Exp 2.

---

## Exp 2 — SAM 3.1 fine-tune as segmenter (LoRA + mmseg)

**Hypothesis**: swap SegFormer for SAM 3.1 in the team's exact pipeline with
LoRA on the encoder.

**Config**: `seg/configs/sam3_lora_windblade_8class.py`

- **Backbone**: `Sam3Backbone` (mmseg wrapper at `seg/sam3_backbone.py`) wrapping
  SAM 3.1's `Sam3DualViTDetNeck` → 4 multi-scale features at 256 ch each
  `[288², 144², 72², 36²]`.
- **Trainable**: PEFT LoRA (r=8, α=16, dropout=0.1) on every `attn.qkv` and
  `attn.proj`; plus the neck convs and the decode head (`SegformerHead`).
  9.4 M of 455 M params trainable (2.06%).
- **Monkey-patch** (`seg/_torch_compat.py`): SAM 3.1's `perflib.fused.addmm_act`
  is inference-only; replaced with a training-aware fallback when
  `torch.is_grad_enabled()`.
- **Crop size**: 1008² (SAM RoPE is fixed at 1008²).
- **Data pipeline**: identical to team's
  `segformer_mit-b4_windblade_8class_blade_filter.py` except crop_size —
  `BladeFilteredRandomCrop` (full), `RareClassCrop` (rare),
  `RandomResize(0.5–2.0)`, `RandomFlip(0.5)`, `PhotoMetricDistortion`,
  `ConcatDataset(full + RepeatDataset×5(rare))`,
  `WindBlade7ClassDataset`, `blade_masks_640`.
- **Optimizer**: AdamW lr=6e-5, wd=0.01, decode_head lr_mult=10× — same as team.
- **Schedule**: LinearLR(500) → PolyLR(eta_min=0, power=1) to 30 k iters.
- **Batch**: 8 (team's 4, bumped because encoder is mostly frozen with LoRA).
- **max_iters**: 30 k (vs team's 300 k — foundation-model fine-tune needs less).
- **Loss**: CrossEntropyLoss (same as team).
- **Test mode**: slide crop 1008, stride 504. TTA flips + multi-scale optional.
- **Val**: `FastValHookV2` on `val_diuid_7.json` every 5 k iters.

**Key departures from team config**:
- Backbone init: ImageNet MiT-B4 → SAM 3.1 (11 M imgs + 1 B masks pretrained)
- Trainable params: ALL ~70 M → LoRA only 9.4 M / 455 M
- max_iters: 300 k → 30 k
- batch_size: 4 → 8
- crop_size: 1024 → 1008
- warmup: 1500 → 500 iters (proportional)
- decode-head `in_channels`: `[64,128,320,512]` → `[256,256,256,256]`

---

## Exp 3 — DINOv2-Large fine-tune as segmenter (LoRA + mmseg)

**Hypothesis**: SAM 3.1 carries 1 B-mask segmentation pretraining; DINOv2 carries
pure self-supervised representation pretraining (LVD-142 M, no masks). Comparing
them on the same recipe isolates how much of SAM's effect comes from the *mask*
pretraining vs the general ViT features.

**Config**: `seg/configs/dinov2_lora_windblade_8class.py`

- **Backbone**: `Dinov2Backbone` (mmseg wrapper at `seg/dinov2_backbone.py`).
  HF `facebook/dinov2-large` (24-layer ViT, hidden 1024, patch 14).
- **Multi-scale neck**: DINOv2 is a plain ViT (single-scale output). We add a
  ViTDet-style `SimpleFPN` (per Li et al. 2022) that takes the final 72² patch
  grid and produces 4 outputs at 256 ch each, scales `[288², 144², 72², 36²]` —
  same shapes/channels as the SAM 3.1 backbone, so the SegformerHead config is
  unchanged.
- **Trainable**: PEFT LoRA (r=8, α=16, dropout=0.1) on every
  `attention.attention.{query,key,value}` and `attention.output.dense` Linear;
  plus the SimpleFPN convs and the decode head. ~10 M of ~310 M params
  trainable (~3.2 %) — slightly higher LoRA-fraction than SAM 3.1 because
  DINOv2 is smaller.
- **Crop size**: 1008² (DINOv2 supports arbitrary multiples of 14; matched to
  SAM for an apples-to-apples comparison).
- **Data pipeline / optimizer / schedule / loss / val hook**: identical to Exp 2.
- **Pretrained weights**: HuggingFace `facebook/dinov2-large` (not gated — no
  HF auth needed, unlike SAM 3.1).

**Key differences vs Exp 2**:
- Pretraining: 1 B masks (SAM) → 142 M images, no masks (DINOv2)
- Param count: 455 M → ~310 M
- Multi-scale neck: native `Sam3DualViTDetNeck` → added `SimpleFPN`
- No `_torch_compat` monkey-patch needed (DINOv2 has no inference-only paths)

---

## Exp 4 — ResNet-152 + DeepLabV3+, 5-class merged (CNN comparison)

**Hypothesis**: CNN baseline (vs the ViT backbones in Exps 2-3) on the same data
pipeline and recipe. Lets us compare backbone families on identical data.

**Config**: `seg/configs/deeplabv3plus_r152_windblade_5class.py`

- **Backbone**: `ResNet` depth=152, pretrained `mmcls://resnet152` (ImageNet),
  all params trainable.
- **Head**: `DepthwiseSeparableASPPHead` (DeepLabV3+), `num_classes=5`.
  Aux FCN head `loss_weight=0.4`.
- **Output**: 5-class merged (BG + 4 defects via `WindBladeMergedDataset` LUT).
- **Crop**: 1024² (matches team — no SAM constraint).
- **Batch**: 4 (team's value).
- **Data pipeline**: identical to team's plus `ApplyMergedLUT` after
  `LoadAnnotations`, `RareClassCrop rare_classes=[2,3,4]` (in merged space).
- **Optimizer / schedule / loss**: same as Exp 2.
- **Train data after WTG leak filter**: 9,522 + 2,507 × 5.

---

## Per-WTG split (DIUID protocol) — leak fixed in our loader

Source `train.json` / `train_rare.json` include images from the same **호기
(WTGs)** that appear in `val_diuid_7.json` and `test_diuid_10.json`:

```
train.json:       10,476 total → 912 leaked  (8.7%)
train_rare.json:   2,699 total → 150 leaked  (5.6%)
val_diuid_7  ∩ test_diuid_10  = 0 WTGs      (val and test are clean)
```

`windblade_merged.py` accepts `exclude_wtg_jsons=[...]` and strips those WTGs
at load time — no regenerated JSONs on disk, easy to add/remove held-out splits.

---

## Decisions deferred

- **SAM 3.1 V2** — text-prompt multi-call segmenter. Next after Exp 2.
- **SAM 3.1 V3** — modify mask decoder. Research-grade, parked.
- **Full SAM 3.1 fine-tune** (vs LoRA). Parked; LoRA is the right starting point.
