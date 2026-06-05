# Wind-blade defect detection — experiments

> 📊 **Live experiments dashboard** → see [`dashboard/`](./dashboard/) (open
> [`dashboard/dmdd_dashboard.html`](./dashboard/dmdd_dashboard.html) for the
> rendered version with bullseye target, per-domain bar charts, Pareto scatter,
> leaderboard, and per-experiment cards).

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
  ( 7 WTGs) / 3,083 test.
  All splits WTG-disjoint (per-WTG leak fixed in `windblade_merged.py`).

---

## Exps 1-4 — backbone exploration

Which pretrained backbone is the right base for SAM-style defect segmentation on
wind-blade imagery? Four backbones tried against the **same data pipeline,
optimizer, schedule, and loss** as the team's SegFormer recipe — only the encoder
and (where needed) the multi-scale neck change between rows.

| # | Backbone | Adaptation | Key choice | Result |
|---|---|---|---|---|
| 1 | SAM 3.1 zero-shot | none — text prompts via `Sam3Processor` | one prompt per class (`"crack on wind turbine blade"` etc.) at thresholds 0.5 and 0.05 | **0 detections** at 0.5; spurious huge masks at 0.05 → SAM 3.1 doesn't understand wind-defect concepts zero-shot. Motivated Exp 2. |
| 2 | **SAM 3.1 — LoRA fine-tune** *(headline recipe — inherited by all later exps)* | PEFT **LoRA r=8 / α=16 / dropout 0.1** on every `attn.qkv` and `attn.proj`; neck + decode head also trainable | 9.4 M of 455 M params trainable (**2.06 %**); 1008² crop (SAM RoPE constraint); 30 k iters / batch 8 / AdamW; `_torch_compat` monkey-patch to make SAM's inference-only `addmm_act` differentiable | Beats team SegFormer baseline at 5 k iters already (mIoU 38 % vs 32 %) — see configs under [`seg/configs/sam3_lora_windblade_8class*.py`](seg/configs/) |
| 3 | DINOv2-Large — LoRA fine-tune | same LoRA + ViTDet `SimpleFPN` (4 outputs at 256 ch, scales `[288²,144²,72²,36²]`) so the SegFormer head is unchanged | ~10 M of ~310 M trainable (**3.2 %**); no monkey-patch needed (no inference-only paths); HF-public weights (no auth wall) | Isolates the value of SAM's 1 B-mask pretraining vs DINOv2's 142 M self-supervised images at matched recipe |
| 4 | ResNet-152 + DeepLabV3+ | full fine-tune (no LoRA) | `DepthwiseSeparableASPPHead`, `num_classes=5` merged; aux FCN head `loss_weight=0.4`; 1024² crop, batch 4 | CNN baseline for the backbone-family A/B; same data pipeline (+ `ApplyMergedLUT` after `LoadAnnotations`) |

**Shared recipe** (Exps 2-4 and inherited by Exps 5-11): AdamW lr = 6e-5,
wd = 0.01, decode-head `lr_mult = 10×`; LinearLR(500) → PolyLR(η_min=0, p=1) to
30 k iters; `BladeFilteredRandomCrop` (full) / `RareClassCrop` (rare) /
`RandomResize(0.5–2.0)` / `RandomFlip(0.5)` / `PhotoMetricDistortion`;
`ConcatDataset(full + RepeatDataset×5(rare))`; slide-crop inference at
stride 504; `FastValHookV2` on `val_diuid_7.json` every 5 k iters.
## Exps 5-11 — production iteration toward the 85 R / 50 P target

Exps 2-4 establish backbone choice; Exps 5-11 are the **production-engineering arc**:
scale the data, fix the loss, oversample backgrounds, calibrate per class, and finally
stack a second network on top of the segmenter to chase precision without losing
recall. Every experiment in this block uses the **SAM 3.1 LoRA** stack from Exp 2 —
only the data mix, loss/sampling, head architecture, or inference-time pipeline
changes, so each step is a clean A/B vs its predecessor.

**Common eval protocol** for all rows below:
- **micro image-level metrics** (R / P): image is TP for class C iff GT has C *and*
  the model predicts ≥ 1 pixel of C; FP iff predicted on an image where GT has none.
  Pooled across the 8 defect classes.
- **per-class softmax threshold T\***: swept on val, pick the largest T\* such that
  per-class P ≥ 0.50, then apply to test (no leakage). "+ T\*" rows below.
- **full-TTA**: 6-view (id + flip × 3 scales), softmax averaged across views before T\*.

**Live numbers + per-class tables**: see [`dashboard/dmdd_dashboard.html`](dashboard/dmdd_dashboard.html)
or [`EXPERIMENTS.md`](EXPERIMENTS.md). Configs referenced by version are under
[`seg/configs/sam3_lora_windblade_8class_v{15,16,17,19,20,21,22}*.py`](seg/configs/).

| # | v | Idea / hypothesis | Key recipe delta | Best result (R / P) | Take-away |
|---|---|---|---|---|---|
| 5 | **v15** | Train **one** model on team + overseas; can a unified backbone beat two specialists? | **CWCE + Dice** loss; tile sampling with `--n-bg=3`; combined 16,123-img train set; warm-start from v1 | **88.0 R / 59.4 P** overseas (full-TTA + T\*) | Clears target on overseas, falls short on team (81 R / 51 P) — too defect-heavy a manifest |
| 6 | **v16** | Upper-bound: Cracks specialist trained **only** on overseas lab labellers | Restricted-scope masks (`la_crack` + `bond_crack`), 2,993-img train | **88.0 R / 73.6 P** on 303-img lab test | What's achievable when label-style + domain match — useful ceiling for the Cracks task |
| 7 | **v17** | v15 over-predicts → starve it of defects: **2× background tile oversampling** | `--n-bg=6` (BG share **56 %** of batch); fixed overseas blade-mask path lookup | **80.1 R / 59.6 P** combined (full-TTA + T\*) | +5–8 pp precision everywhere, but recall slips → motivates v20's multi-task head |
| 8 | **v19** | Isolate **labeller-style effect** vs v16: train v16-recipe model on remasked **team** Cracks | 531-img team Crack-only subset, masks zeroed outside `la_crack`/`bond_crack` | 78.3 R / 69.1 P on 60-img team Crack test | **+10 pp recall gap** vs v16 at identical scope ⇒ label style alone drives a sizable portion of cross-domain Cracks gap |
| 9 | **v20** | Add a **second task**: tile-level binary aux head jointly trained with the dense seg head | New `SegformerHeadWithAux` (GAP → MLP → 1 logit); `L = CWCE + Dice + 0.1 × BCE(aux, has_defect)`; warm-start from v17 | **84.9 R / 50.7 P** combined (no-TTA + T\*) — closest any combined-data model has come to target | Aux head regularises the encoder *even with the logit dropped at inference*. **Best ship candidate.** Gating modes A/B (soft × / hard gate) pending — should lift precision further |
| 10 | **v21** | Two-stage **cascade**: train a tile-level binary classifier as stage-1, gate v15 / v17's per-tile output | EfficientNet-B0 (5 M, ImageNet) · BCE + `pos_weight`; sweep hard-gate `T_gate ∈ {0.1…0.9}` × {soft ×, hard gate} per domain | best **82.5 R / 36.6 P** (v15 + v21, team, hard T=0.9) | Stage-1 ranker too weak (AUROC 0.77, AUPR 0.07 on 2.7 %-positive tiles): +5 pp P costs 5-8 pp R, **net F1 worse** than seg alone — useful negative result |
| 11 | **v22** | v21 ablation: swap CNN stage-1 for **ViT** (Swin-T, 28 M params) — does attention help on the imbalanced tile-binary task? | Swin-Tiny (timm, ImageNet) · same loss / data / iters as v21 | best **73.5 R / 44.7 P** team, 80.3 R / 55.9 P overseas (v17 + v22) | Bigger ViT lifts P further but at much harsher R cost; **cascade is out-classed by v20's multi-task head** end-to-end |

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
