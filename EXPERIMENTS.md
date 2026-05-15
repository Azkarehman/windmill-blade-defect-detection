# DMDD experiments — SAM 3.1 segmenter swap

## Project context

- **Goal**: wind-turbine blade defect semantic segmentation. Customer team's
  production baseline = SegFormer-B4 (9-class), mIoU 32.35% on `test_diuid_17wtg`,
  recall 0.79 / precision 0.57 (sample-weighted micro). Customer target = **micro
  image-level recall ≥ 85%**.
- **Approach in this repo**: swap SegFormer for **SAM 3.1** (Meta, March 2026 —
  848M params, pretrained on 11M images + 1B masks) as a drop-in mmseg backbone,
  fine-tune via LoRA on attention layers. Same data pipeline, same optimizer,
  same loss — only the backbone changes.
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
`Sam3Processor`, score threshold 0.5 then 0.05 diagnostic. 10 defect-heavy test
images.

**Results**:
- Threshold 0.5: **0 detections** across all 10 images and all 4 classes.
- Threshold 0.05: a few low-confidence detections (max 0.151), masks cover huge
  non-defect regions (sky, blade body).

**Takeaway**: SAM 3.1 does NOT understand wind-turbine defect concepts
zero-shot. Fine-tune required.

---

## Exp 2 — SAM 3.1 fine-tune as segmenter (LoRA + mmseg) — HEADLINE

**Hypothesis**: "swap SegFormer for SAM 3.1 in the team's exact pipeline" with
LoRA on the encoder. Should beat the SegFormer baseline on the same data.

**Config**: `seg/configs/sam3_lora_windblade_8class.py`

- **Backbone**: `Sam3Backbone` (mmseg wrapper at `seg/sam3_backbone.py`) wrapping
  SAM 3.1's `Sam3DualViTDetNeck` → 4 multi-scale features at 256 ch each
  `[288², 144², 72², 36²]`.
- **Trainable**: PEFT LoRA (r=8, α=16, dropout=0.1) on every `attn.qkv` and
  `attn.proj`; plus the neck convs and the decode head (`SegformerHead`).
  **9.4 M of 455 M params trainable (2.06%)**.
- **Monkey-patch** (`seg/_torch_compat.py`): SAM 3.1's `perflib.fused.addmm_act`
  is inference-only; replaced with a training-aware fallback when
  `torch.is_grad_enabled()`.
- **Crop size**: 1008² (SAM RoPE is fixed at 1008²).
- **Data pipeline**: **identical to team's**
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

**Smoke (50 iters)**: 1.6 s/iter, batch 8, 22 GB peak GPU mem.
Loss 2.13 → 1.93, train acc 24% → 96% (mostly BG winning early).

**Results**:
| iter  | val mIoU | val mean recall (macro) | val recall (micro) |
|-------|----------|--------------------------|---------------------|
| 5 k   | **38.09%** | 57.60%                 | ~69.1%              |
| 10 k  | pending  | pending                  | pending             |
| 30 k  | pending (TTA inference in progress) | pending | pending |

5 k mIoU already above team baseline (32.35%); recall still below.

**Key departures from team config**:
- Backbone init: ImageNet MiT-B4 → SAM 3.1 (11 M imgs + 1 B masks pretrained)
- Trainable params: ALL ~70 M → LoRA only 9.4 M / 455 M
- max_iters: 300 k → 30 k
- batch_size: 4 → 8
- crop_size: 1024 → 1008
- warmup: 1500 → 500 iters (proportional)
- decode-head `in_channels`: `[64,128,320,512]` → `[256,256,256,256]`

---

## Exp 3 — ResNet-152 + DeepLabV3+, 5-class merged (CNN comparison)

**Hypothesis**: CNN baseline (vs SAM 3.1 transformer) on the same data pipeline
and recipe. Lets us compare backbone families on identical data.

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

**Smoke (20 iters)**: 2.2 s/iter, ~52 GB GPU mem (while SAM 3.1 also running),
loss 2.25 → 2.22. ResNet-152 ImageNet weights load cleanly.

**Train data after WTG leak filter**: 9,522 + 2,507 × 5.

**Results**: pending — 30 k iter run kicked off 2026-05-13, parallel with Exp 2.

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
