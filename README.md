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

## Exps 5-11 — production iteration toward the 85R / 50P target

Exps 2-4 establish the backbone and recipe. Exps 5-11 are the production-engineering
arc: scale the data, fix the loss, calibrate per class, and finally add a second
network on top of the segmenter to chase precision without losing recall. Every
experiment in this block uses the **SAM 3.1 LoRA** stack from Exp 2 — only the data
mix, loss/sampling, head architecture, or inference-time pipeline changes between
versions, so each step is a clean A/B against its predecessor.

A common evaluation protocol is used throughout:
- **micro image-level metrics** (R / P): for each defect class, image is TP iff the
  GT contains that class AND the model predicts at least one pixel of it; FP iff
  the model predicts the class on an image where GT has none. Pooled across all
  defect classes.
- **per-class softmax threshold T\*** ("calibrated thresholds"): we sweep the
  per-class softmax threshold on the **val** set and pick the largest T\* such that
  per-class precision ≥ 0.50, then apply that T\* on the **test** set (no leakage).
  This is the "+ T\*" rows below.
- **full-TTA**: 6-view test-time augmentation (id + flip × 3 scales). Averaged
  softmax across views before threshold.

---

## Exp 5 — v15 — combined team + overseas training

**Hypothesis**: a single SAM 3.1 LoRA model trained on **team + overseas
together** generalises across both domains better than two specialists, even
though the two datasets come from different cameras (NIKON Z 7 vs M3E drone),
different labeller groups, and different blade designs.

**Config**: `seg/configs/sam3_lora_windblade_8class_v15_combined_noleak.py`

- **Backbone / LoRA / optimizer / schedule**: identical to Exp 2.
- **Data**: 9,564 team `train_no_leak` + 6,559 overseas (lab + trained labellers
  merged) = **16,123 images**. Evaluated separately on the team
  `test_diuid_10` (3,083) and the overseas test split (720).
- **Loss**: **CWCE + Dice** (class-weighted cross-entropy + Dice; replaces the
  vanilla CE used in Exp 2). The CWCE weights are computed once from the
  combined train set's inverse class-pixel frequency, clipped to [0.5, 25].
- **Tile sampling** (new): rather than crop-from-image, we operate on the
  team's tile manifest with `--n-bg=3` — three random background-only 1008²
  tiles per defect tile, so each batch has known foreground/background mix.
  49,398 team + 32,111 overseas tiles per epoch.
- **Warm-start**: SAM 3.1 backbone + LoRA initialised from `v1`'s
  `iter_30000.pth` so the recipe converges in 30 k iters even with the larger
  data pool.

**Results**:

| variant | mIoU | micro R | micro P | hits 85R/50P |
|---|---|---|---|---|
| team test, no-TTA + T\*       | 0.296 | 80.89% | 44.54% |  |
| team test, full-TTA + T\*     | 0.304 | 81.10% | 51.11% |  |
| **overseas test, full-TTA + T\*** | **0.325** | **87.96%** | **59.37%** | ✓ |
| COMBINED, full-TTA + T\*      | —     | 83.26% | 53.01% |  |

**Takeaway**: combined-data training clears the customer target on the overseas
test set (88R/59P) but falls short on team test (81R/51P) — the team test
distribution carries multi-label / no-defect cases that the overseas-heavy
mixture mis-calibrates. Motivates v17's BG-tile oversampling.

---

## Exp 6 — v16 — overseas-lab Cracks specialist

**Hypothesis**: the overseas dataset is labelled by two distinct groups — a
"lab" group (large pixel-area Cracks, 3 641 imgs) and a "trained" group (81
small Crack imgs + 6 other task types). Train **only on lab-labelled Crack
images** to test what a Cracks specialist looks like on its native
distribution. Also gives an apples-to-apples baseline that Exp 8 (v19) will
challenge with team Cracks only.

**Config**: `seg/configs/sam3_lora_windblade_8class_v16_lab_only.py`

- **Data**: overseas lab labellers only — 2 993 train / 345 val / 303 test.
- **Scope**: classes 3 (`la_crack`) and 5 (`bond_crack`); all other class IDs
  are zeroed in the masks before loading.
- **Recipe**: CWCE + Dice + LoRA-r8, `--n-bg=3`, 30 k iters DDP-2 — same as
  v15, just on the restricted scope.

**Results** (303-image lab Cracks test):

| variant | mIoU | micro R | micro P |
|---|---|---|---|
| no-TTA baseline    | **0.581** | 92.47% | 72.58% |
| full-TTA + T\*     | 0.593 | 88.01% | 73.64% |

**Takeaway**: a single-domain Cracks specialist hits **88R / 73P** — well past
the customer target — but only on its own labeller's distribution. Doesn't
generalise to team Cracks (see Exp 8). Useful as the upper-bound reference
for what's achievable when label style and distribution are matched.

---

## Exp 7 — v17 — combined + 2× background tile sampling

**Hypothesis**: v15 has high recall but weak precision on team test because the
tile manifest is too defect-heavy; the model rarely sees clean-blade tiles and
over-predicts. **Doubling the background-only tile ratio** (`--n-bg=6` vs
v15's `3`) should reduce FPs without sacrificing recall.

**Config**: `seg/configs/sam3_lora_windblade_8class_v17_combined_2xbg.py`

- **Data**: same train images as v15, but the tile sampler emits 46 616 defect
  + **60 067 BG** tiles per epoch (BG share **56 %** of each batch).
- **Bug fix**: patched `load_blade_mask` to resolve overseas mask paths
  consistently — v15 silently fell back to all-blade masks on overseas tiles.
- **Recipe**: otherwise identical to v15.

**Results**:

| variant | mIoU | micro R | micro P | Δ vs v15 |
|---|---|---|---|---|
| team test, no-TTA + T\*       | 0.341 | 80.42% | **52.08%** | +7.5 pp P |
| team test, full-TTA + T\*     | 0.348 | 79.27% | **58.65%** | +7.5 pp P |
| overseas test, full-TTA + T\* | 0.347 | 81.94% | 63.92% | +4.6 pp P |
| COMBINED, full-TTA + T\*      | —     | 80.11% | 59.55% | +6.5 pp P |

**Takeaway**: BG oversampling buys **+5–8 pp precision** on every test set at
the cost of ~3 pp recall. Precision crosses the 50 % target on both domains,
but recall now falls short on team. The recall/precision tug-of-war
motivates v20's multi-task head — get the precision lift from a separate
classifier signal rather than from sampling bias.

---

## Exp 8 — v19 — team Cracks remasked (apples-to-apples vs v16)

**Hypothesis**: v16's 88 R / 73 P comes from either (a) the lab labellers' style,
or (b) the data distribution being easier. Train a v16-shaped model on **team
Cracks only** at matched class scope and ratio: same model, same recipe, same
scope — only the labeller changes.

**Config**: `seg/configs/sam3_lora_windblade_8class_v19_team_cracks_apples.py`

- **Data**: team images containing Cracks, masks **remasked** so only
  `la_crack` and `bond_crack` pixels survive (everything else zeroed). 531
  train / 44 val / 60 test — matched scale to v16 for a clean comparison.
- **Recipe**: CWCE + Dice + LoRA-r8, `--n-bg=3`, 30 k iters DDP-2.

**Results** (60-image team Crack-only test):

| variant | mIoU | micro R | micro P |
|---|---|---|---|
| no-TTA + T\* | 0.528 | 78.33% | 69.12% |

**Takeaway**: team labellers' Cracks cap at **78R / 69P**, vs v16's overseas-lab
Cracks at **88R / 73P** — a **+10 pp recall** gap purely from labeller style at
identical scope and model. This is the cleanest evidence in the project that
*label style* (not data distribution) drives a sizable fraction of the gap
across labelling groups, and the result that justifies investing in tighter
team-labeller consistency rather than only chasing model improvements.

---

## Exp 9 — v20 — multi-task model with auxiliary tile-classification head

**Hypothesis**: precision can be improved without harming recall by giving the
model a **second task**: predict per-tile *"does this tile contain any defect
pixel?"* alongside the dense segmentation. The aux head acts as a regulariser
during training, and at inference its logit can multiplicatively gate or
re-rank the per-pixel softmax — turning a single network into an implicit
two-stage classifier-then-segmenter.

**Config**: `seg/configs/sam3_lora_windblade_8class_v20_aux_cls.py`

- **Head** (`seg/segformer_head_with_aux.py`): existing `SegformerHead` (dense
  branch) **plus a new aux branch**: global-average-pool over the deepest
  encoder feature → 2-layer MLP → 1 logit. The aux branch shares all encoder
  + neck weights with the seg branch.
- **Loss**:
  `L = L_CWCE + L_Dice + 0.1 × BCE(aux_logit, y_aux)` where
  `y_aux = (mask contains any defect pixel)`. The 0.1 weight keeps the aux
  task a regulariser; we never let it dominate the dense gradient.
- **Inference modes** (planned):
  - *Mode A — soft multiply*: scale per-pixel softmax by `sigmoid(aux_logit)`
    so empty-looking tiles can't fire confidently.
  - *Mode B — hard gate*: drop a tile's seg output entirely if
    `sigmoid(aux_logit) < T_abstain`.
- **Warm-start**: v17 `iter_30000.pth` — v20 is "v17 + aux head" continued for
  30 k more iters.

**Results** (aux head currently silently discarded at inference; gating modes
A/B pending):

| variant | mIoU | micro R | micro P | hits 85R/50P |
|---|---|---|---|---|
| team test, no-TTA + T\*       | 0.301 | 83.74% | 48.47% |  |
| **overseas test, no-TTA + T\*** | **0.312** | **87.37%** | **59.80%** | ✓ |
| **COMBINED, no-TTA + T\***   | —     | **84.89%** | **50.69%** | ✓ |

**Takeaway**: v20 reaches **84.89 R / 50.69 P on combined test** — the closest
any combined-data model has come to the customer target without trading
recall for precision (v17 was 80.11R / 59.55P). The aux head appears to
regularise the encoder even when its logit is dropped at inference; turning
on the gating modes is expected to lift precision further. v20 is currently
the **best ship candidate** of all combined-data models.

---

## Exp 10 — v21 — EfficientNet-B0 cascade stage-1 classifier

**Hypothesis**: a separately-trained tile-level binary classifier ("is this
tile defective?") can act as **stage 1 of a cascade**, gating the v15/v17
segmenter's output to drop predictions on empty tiles. This decouples the
recall-driver (the seg model, kept loose) from the precision-driver (a
specialist binary classifier with its own threshold).

**Config**: training under `seg/configs/sam3_lora_v21_*` family; inference
orchestrated by `eval_v21_cascade.py`.

- **Backbone**: EfficientNet-B0 (torchvision, ImageNet-pretrained, 5 M params).
- **Head**: GAP → Dropout(0.2) → Linear(1280, 1).
- **Loss**: BCE-with-logits with `pos_weight = (#neg / #pos)` to counter the
  severe tile-level imbalance (2.7 % defect tiles).
- **Data**: v17's tile manifest, treated as 0/1 (defect / BG).
- **Iters**: 20 k DDP-2.
- **Cascade modes**:
  - *soft multiply*: `output_softmax × sigmoid(tile_score)`, drop CCs whose
    product < 0.05.
  - *hard gate*: discard the entire tile if `sigmoid(tile_score) < T_gate`.
    Sweep `T_gate ∈ {0.1, …, 0.9}`, pick the best (mode, T_gate) by F1 per
    domain.

**Stage-1 standalone (tile-level)**:

| AUROC | AUPR | P @ 0.5 | R @ 0.5 |
|---|---|---|---|
| 0.768 | 0.072 | 82.1% | 5.6% |

**Cascade results** (best operating point per pair):

| seg + cls pair | test set | baseline R / P | best (mode / T_gate) | gated R / P |
|---|---|---|---|---|
| v17 + v21 | team     | 87.74 / 30.51 | hard / 0.9 | 80.28 / **35.62** |
| v17 + v21 | overseas | 82.53 / 53.68 | hard / 0.3 | 80.76 / **55.11** |
| v15 + v21 | team     | 90.25 / 29.60 | hard / 0.9 | 82.51 / **36.62** |
| v15 + v21 | overseas | 84.88 / 49.96 | hard / 0.8 | 77.09 / **55.26** |

**Takeaway**: cascade gating gives a **modest +5 pp precision** lift at the
cost of **5-8 pp recall**. The stage-1 classifier is too weak a ranker
(AUROC 0.77, AUPR 0.07) given the imbalance — once you raise `T_gate` high
enough to clean up FPs, you also kill real positives. Net F1 slightly
*worse* than v17/v15 alone. A useful negative result: the cascade idea
itself is sound, but stage-1 needs a stronger backbone — motivates v22.

---

## Exp 11 — v22 — Swin-T cascade stage-1 (ViT alternative)

**Hypothesis**: replace v21's CNN stage-1 with a **ViT** (Swin-Tiny) to see
whether tile-level binary defect detection benefits from attention-based
global context rather than EfficientNet's local convolutions. Direct A/B vs
v21 — same training data, same recipe, only the backbone changes.

- **Backbone**: Swin-Tiny (timm, ImageNet-pretrained, ~28 M params; 5.6× v21's
  parameter count).
- **Head / loss / data / iters**: identical to v21.

**Cascade results** (best operating point per pair):

| seg + cls pair | test set | baseline R / P | best (mode / T_gate) | gated R / P |
|---|---|---|---|---|
| v17 + v22 | team     | 87.94 / 29.75 | hard / 0.9 | **73.51** / 44.72 |
| v17 + v22 | overseas | 82.97 / 52.85 | hard / 0.4 | 80.32 / 55.93 |

**Takeaway**: Swin-T cascade lifts team-test precision *more* than v21
(44.7 % vs 35.6 %) — but at a **much harsher recall cost** (73.5 % vs 80.3 %).
ViT didn't dramatically improve ranking quality for the imbalanced tile-binary
task; if anything, the bigger backbone overfit the easy negatives. Together
with v21 this is the project's clearest signal that a *cascade* approach is
out-classed by the *multi-task* approach (v20) — a single network with an
aux classification head reaches 85R / 50P territory in one forward pass, while
the two-stage cascade caps below it on team test and adds inference complexity.

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
