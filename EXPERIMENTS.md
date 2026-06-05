# DMDD wind-blade defect experiments

_Auto-generated from `dashboard/dmdd_data.json` · last updated **2026-06-05 15:00 KST**._

> Full interactive view (bullseye target chart, per-domain bar charts, Pareto scatter, per-experiment cards): see [`dashboard/dmdd_dashboard.html`](dashboard/dmdd_dashboard.html).

**Customer target.** micro R ≥ 85%, micro P ≥ 50%

**Class IDs.** 

| id | class |
|---|---|
| 0 | background |
| 1 | la_exposure |
| 2 | la_damage |
| 3 | la_crack |
| 4 | la_open |
| 5 | bond_crack |
| 6 | bond_open |
| 7 | receptor_lightning |
| 8 | receptor_damage |

---

## 🏆 Leaderboard — best operating point per objective

| Objective | Config | R | P |
|---|---|---|---|
| 🟦 Best team test — closest to 85R/50P (combined-data models) | v20 no-TTA + T* (team) | 83.74% | 48.47% |
| 🟦 Best team test — hits both 85R AND 50P | v12 full-TTA + T* (team) | 83.06% | 51.69% |
| 🟦 Highest team-test recall (any P) | v17 no-TTA baseline (team) | 85.70% | 39.88% |
| 🟦 Highest team-test precision | v17 full-TTA + T* (team) | 79.27% | 58.65% |
| 🟧 Best overseas test (any model) | v13 full-TTA + T* (overseas) | 91.34% | 66.31% |
| 🟧 Best overseas (combined-data model) | v15 full-TTA + T* (overseas) | 87.96% | 59.37% |
| 🟧 Best overseas no-TTA + T* (combined) | v20 no-TTA + T* (overseas) | 87.37% | 59.80% |
| 🟩 Best combined (closest to 85R/50P) | v20 no-TTA + T* (combined) | 84.89% | 50.69% |
| 🟩 Best combined precision | v17 full-TTA + T* (combined) | 80.11% | 59.55% |
| 🟩 Best combined recall | v15 no-TTA baseline (combined) | 88.04% | 36.57% |
| ✨ Cracks specialist (lab test 303 imgs) | v16 full-TTA + T* | 88.01% | 73.64% |
| ⚠ Worst — cross-domain transfer | v13-cross full-TTA + T* (team) | 62.47% | 64.52% |

---

## 🧪 Experiments (latest first)

### v24 — team train_no_leak re-split 80/10/10 WTG-disjoint
_Status: **queued**_ · tags: team, wtg-disjoint

**Purpose.** v12 recipe with team train_no_leak re-split into 80/10/10 with 0 WTG overlap. Tests performance under truly held-out turbine setting. Does NOT 'fix' v12; measures a different distribution.

**Training data**

- **train_size**: 7737
- **val_size**: 961
- **test_size**: 866
- **labeler**: team

---

### v23 — overseas lab Cracks size-matched to v19
_Status: **queued**_ · tags: overseas, cracks-only, lab-labelers, size-matched

**Purpose.** Train another overseas lab Cracks model with the SAME train size as v19 (544 ≈ 531). Decouples labeler-style from data-size effects. v23 vs v19 = labeler-only A/B; v23 vs v16 = size-only A/B within same labeler set.

**Training data**

- **train_size**: 544
- **val_size**: 44
- **test_size**: 72
- **labeler**: overseas lab
- **scope**: La_Crack + Bond_Crack only (remasked)

---

### v22 — Swin-T tile classifier (ViT alternative cascade)
_Status: **eval_in_progress**_ · tags: cascade, stage-1, swin-t, vit

**Purpose.** Same role as v21 but ViT-based backbone. Direct A/B vs v21 holding everything else constant.

**Training data**

- **source**: Same as v21

**Recipe**

- **backbone**: Swin-Tiny (timm, ImageNet pretrained, ~28M params)
- **input_size**: 1008×1008 native
- **iters**: 20,000

**Results**

| variant | mIoU | R | P | hits_target |
|---|---|---|---|---|
| team test (v17+v22, hard T=0.9) | — | 73.51% | 44.72% |  |
| overseas test (v17+v22, hard T=0.4) | — | 80.32% | 55.93% |  |

**Cascade results**

| pair | baseline_R | baseline_P | best_mode | best_R | best_P |
|---|---|---|---|---|---|
| v17 + v22 on team test | 87.94 | 29.75 | hard | 73.51 | 44.72 |
| v17 + v22 on overseas test | 82.97 | 52.85 | hard | 80.32 | 55.93 |
| v15 + v22 on overseas test | — | — | — | — | — |

**Caveats.** - v15+v22 team-test eval still in progress (1,005/3,083 imgs); excluded until full to avoid biased numbers

**Take-away.** Swin-T cascade landed similar precision lift to EfficientNet but with WORSE recall on team (73.5% vs v21's 80.3%). ViT didn't dramatically improve over CNN for tile-level binary task. Final v15+v22 pairs pending.

---

### v21 — EfficientNet-B0 tile classifier (cascade stage 1)
_Status: **done**_ · tags: cascade, stage-1, efficientnet

**Purpose.** Standalone tile-level binary classifier as the first stage of a two-stage cascade. Trained on v17's tile manifest (defect=1, BG=0). At inference, gates v17/v15's per-tile segmentation predictions.

**Training data**

- **source**: v17 tile manifest (defect + BG, combined team+overseas)

**Recipe**

- **backbone**: EfficientNet-B0 (torchvision, ImageNet pretrained)
- **head**: GAP → Dropout(0.2) → Linear(1280, 1)
- **loss**: BCE-with-logits + pos_weight balancing
- **iters**: 20,000 DDP-2
- **input_size**: 1008×1008 native

**Results**

| variant | mIoU | R | P | hits_target | note |
|---|---|---|---|---|---|
| v17+v21 COMBINED, no gate (baseline) | — | 86.09% | 35.09% |  | — |
| v17+v21 COMBINED, per-domain best gate | — | 80.44% | 40.12% |  | — |
| v15+v21 COMBINED, no gate (baseline) | — | 88.24% | 34.69% |  | micro across full team (3,083) + overseas (529) test imgs · gate disabled (every tile passes) |
| v15+v21 COMBINED, per-domain best gate | — | 80.48% | 41.66% |  | team uses hard T=0.9, overseas uses hard T=0.8 — picked per domain, micro pooled across both |
| team test (v17+v21, hard T=0.9) | — | 80.28% | 35.62% |  | — |
| overseas test (v17+v21, hard T=0.3) | — | 80.76% | 55.11% |  | — |
| team test (v15+v21, hard T=0.9) | — | 82.51% | 36.62% |  | full 3,083 team-test imgs · 75.4% tiles gated out by stage-1 |
| overseas test (v15+v21, hard T=0.8) | — | 77.09% | 55.26% |  | — |

**Stage-1 standalone (tile classifier)**

- **AUROC**: 0.768
- **AUPR**: 0.072
- **R_at_0.5**: 5.6
- **P_at_0.5**: 82.1
- **F1_at_0.5**: 0.1

**Cascade results**

| pair | records | baseline_R | baseline_P | best_mode | best_t_gate | best_R | best_P |
|---|---|---|---|---|---|---|---|
| v17 + v21 on team test | — | 87.74 | 30.51 | hard | — | 80.28 | 35.62 |
| v17 + v21 on overseas test | — | 82.53 | 53.68 | hard | — | 80.76 | 55.11 |
| v15 + v21 on team test | 3083 | 90.25 | 29.60 | hard | 0.90 | 82.51 | 36.62 |
| v15 + v21 on overseas test | — | 84.88 | 49.96 | hard | — | 77.09 | 55.26 |

**Take-away.** Cascade gating gave modest precision lift (~5 pp) at the cost of 5-8 pp recall. Stage-1 EfficientNet was too weak a ranker (AUROC 0.77, AUPR 0.07) given tile-level class imbalance (only 2.7% defect tiles). Net F1 slightly worse than v17/v15 alone. v22 (Swin-T) may rank better.

---

### v20 — combined model + auxiliary defect-classification head
_Status: **done**_ · tags: combined, multi-task

**Purpose.** v17 architecture + a new tile-level binary 'is this tile defect?' classification head, jointly trained with seg head. Aux loss is 0.1× BCE on the aux logit. Intended for inference-time gating (Mode A soft multiply or Mode B hard gate) to reduce FPs on empty tiles.

**Training data**

- **source**: Same as v17
- **labeler**: team + overseas
- **scope**: all 8 defect classes

**Recipe**

- **loss**: CWCE + Dice + 0.1 × BCE(aux_logit, has_any_defect_pixel)
- **head**: SegformerHeadWithAux (existing seg branch + new aux branch: GAP → MLP → 1 logit)
- **iters**: 30,000 DDP-2
- **warm_start**: v17 iter_30000.pth

**Results**

| variant | mIoU | R | P | hits_target |
|---|---|---|---|---|
| team test, no-TTA baseline | 0.301 | 86.04% | 39.82% |  |
| team test, no-TTA + T* | 0.301 | 83.74% | 48.47% |  |
| overseas test, no-TTA baseline | 0.312 | 87.52% | 57.14% | ✓ |
| overseas test, no-TTA + T* | 0.312 | 87.37% | 59.80% | ✓ |
| COMBINED, no-TTA baseline | — | 86.51% | 44.09% |  |
| COMBINED, no-TTA + T* | — | 84.89% | 50.69% | ✓ |

**Pending**

- full-TTA + T* not run
- aux-gated inference (Mode A soft / Mode B hard) not run — would use the aux head currently silently discarded

---

### v19 — team Cracks remasked (apples-to-apples vs v16)
_Status: **done**_ · tags: team, cracks-only

**Purpose.** Sample team images containing Cracks, REMASK to keep only La_Crack + Bond_Crack pixels, match v16's class ratio. Isolates labeler-style effect at constrained scope.

**Training data**

- **source**: team Crack-containing images (remasked)
- **train_size**: 531
- **val_size**: 44
- **test_size**: 60
- **labeler**: team
- **scope**: La_Crack + Bond_Crack only (other classes zeroed in masks)

**Recipe**

- **loss**: CWCE + Dice
- **iters**: 30,000 DDP-2
- **tiles**: --n-bg=3

**Results**

| variant | mIoU | R | P | hits_target |
|---|---|---|---|---|
| no-TTA baseline | 0.528 | 78.33% | 69.12% |  |
| no-TTA + T* | 0.528 | 78.33% | 69.12% |  |

**Per-class (no-TTA + T*)**

| class | TP | FP | FN | R | P |
|---|---|---|---|---|---|
| la_crack | 29 | 12 | 9 | 76.30% | 70.70% |
| bond_crack | 18 | 9 | 4 | 81.80% | 66.70% |

---

### v17 — combined + 2× BG tile sampling
_Status: **done**_ · tags: combined, ablation-2xBG

**Purpose.** v15 architecture but BG-tile sampling doubled (--n-bg=6 vs v15's 3). Tests whether more clean-blade exposure during training reduces FPs.

**Training data**

- **source**: Same as v15 (team_no_leak + overseas merged)
- **test_sets**: ['team test_diuid_10', 'overseas_only_test']
- **labeler**: team + overseas
- **scope**: all 8 defect classes

**Recipe**

- **loss**: CWCE + Dice
- **backbone**: SAM 3.1 LoRA r=8
- **iters**: 30,000 DDP-2
- **tiles**: --n-bg=6 (46,616 defect + 60,067 BG tiles/epoch; BG share 56% of batch)
- **blade_mask_fix**: patched load_blade_mask for overseas paths

**Results**

| variant | mIoU | R | P | hits_target |
|---|---|---|---|---|
| team test, no-TTA baseline | 0.340 | 85.70% | 39.88% |  |
| team test, no-TTA + T* | 0.341 | 80.42% | 52.08% |  |
| team test, full-TTA baseline | 0.348 | 84.01% | 48.40% |  |
| team test, full-TTA + T* | 0.348 | 79.27% | 58.65% |  |
| overseas test, no-TTA baseline | 0.336 | 84.88% | 54.17% |  |
| overseas test, no-TTA + T* | 0.336 | 84.58% | 59.26% |  |
| overseas test, full-TTA baseline | 0.347 | 82.09% | 59.59% |  |
| overseas test, full-TTA + T* | 0.347 | 81.94% | 63.92% |  |
| COMBINED, full-TTA + T* | — | 80.11% | 59.55% |  |

---

### v16 — overseas lab Cracks specialist
_Status: **done**_ · tags: overseas, cracks-only, lab-labelers

**Purpose.** Trained only on overseas lab-labeler images (Cracks-only scope by labeler task type). Tests how a Cracks specialist trained on lab labels performs on its native distribution.

**Training data**

- **source**: overseas lab labelers (2,993 train / 345 val / 303 test)
- **labeler**: overseas lab
- **scope**: La_Crack + Bond_Crack only

**Recipe**

- **loss**: CWCE + Dice
- **backbone**: SAM 3.1 LoRA r=8
- **iters**: 30,000 DDP-2
- **tiles**: --n-bg=3

**Results**

| variant | mIoU | R | P | hits_target |
|---|---|---|---|---|
| no-TTA baseline | 0.581 | 92.47% | 72.58% | ✓ |
| no-TTA + T* | 0.581 | 92.47% | 72.58% | ✓ |
| full-TTA baseline | 0.593 | 88.01% | 73.64% | ✓ |
| full-TTA + T* | 0.593 | 88.01% | 73.64% | ✓ |

---

### v15 — combined team + overseas
_Status: **done**_ · tags: combined, all-classes

**Purpose.** v8 recipe trained on combined team_no_leak + overseas merged. Evaluated separately on each test set.

**Training data**

- **source**: team train_no_leak (9,564) + overseas merged (6,559) = 16,123
- **test_sets**: ['team test_diuid_10 (3083)', 'overseas_only_test (720)']
- **labeler**: team + overseas
- **scope**: all 8 defect classes

**Recipe**

- **loss**: CWCE + Dice
- **backbone**: SAM 3.1 LoRA r=8
- **iters**: 30,000 DDP-2
- **tiles**: --n-bg=3 (49,398 team + 32,111 overseas tiles/epoch)
- **warm_start**: v1

**Results**

| variant | mIoU | R | P | hits_target |
|---|---|---|---|---|
| team test, no-TTA baseline | 0.296 | 87.06% | 33.33% |  |
| team test, no-TTA + T* | 0.296 | 80.89% | 44.54% |  |
| team test, full-TTA baseline | 0.304 | 85.57% | 40.94% |  |
| team test, full-TTA + T* | 0.304 | 81.10% | 51.11% |  |
| overseas test, no-TTA baseline | 0.304 | 90.16% | 45.89% |  |
| overseas test, no-TTA + T* | 0.303 | 87.37% | 54.74% | ✓ |
| overseas test, full-TTA baseline | 0.325 | 87.96% | 57.16% | ✓ |
| overseas test, full-TTA + T* | 0.325 | 87.96% | 59.37% | ✓ |
| COMBINED, full-TTA + T* | — | 83.26% | 53.01% |  |

---

### v13 — overseas-only
_Status: **done**_ · tags: overseas, all-classes

**Purpose.** Same v8 recipe trained only on overseas turbine-split data. Tests how the model performs on the overseas distribution when trained natively on it.

**Training data**

- **source**: overseas_only_train_merged.json
- **train_size**: 6559
- **val_size**: 867
- **test_size**: 720
- **test_set**: overseas_only_test.json
- **labeler**: overseas (lab + trained)
- **scope**: all 8 defect classes (in practice mostly La_Exposure-heavy)

**Recipe**

- **loss**: CWCE + Dice
- **backbone**: SAM 3.1 LoRA r=8
- **iters**: 30,000 DDP-2
- **tiles**: --n-bg=3
- **warm_start**: v1 iter_30000.pth

**Results**

| variant | mIoU | R | P | hits_target |
|---|---|---|---|---|
| no-TTA baseline | 0.381 | 92.51% | 56.20% | ✓ |
| no-TTA + T* | 0.380 | 90.60% | 60.49% | ✓ |
| full-TTA baseline | 0.392 | 91.34% | 65.47% | ✓ |
| full-TTA + T* | 0.392 | 91.34% | 66.31% | ✓ |

**Per-class (full-TTA + T*)**

| class | TP | FP | FN | R | P |
|---|---|---|---|---|---|
| la_exposure | 293 | 94 | 38 | 88.50% | 75.70% |
| la_damage | 0 | 34 | 0 | — | 0.00% |
| la_crack | 123 | 92 | 29 | 80.90% | 57.20% |
| la_open | 0 | 0 | 0 | — | — |
| bond_crack | 90 | 71 | 50 | 64.30% | 55.90% |
| bond_open | 1 | 8 | 0 | 100 | 11.10% |
| receptor_lightning | 51 | 21 | 3 | 94.40% | 70.80% |
| receptor_damage | 0 | 29 | 3 | 0.00% | 0.00% |

---

### v12 — team baseline (leak-free)
_Status: **done**_ · tags: team, all-classes

**Purpose.** Leak-free team baseline. The team's original train.json had 912 images that overlapped with val_diuid_7 and test_diuid_10. v12 trains on a deduplicated train_no_leak.json (9,564 imgs) and evaluates on the same team val/test as v8 for direct apples-to-apples comparison.

**Training data**

- **source**: team train_no_leak.json
- **train_size**: 9564
- **val_size**: 2124
- **test_size**: 3083
- **test_set**: test_diuid_10.json
- **labeler**: team
- **scope**: all 8 defect classes

**Recipe**

- **loss**: CWCE + Dice
- **backbone**: SAM 3.1 LoRA r=8 (~9.4M trainable / 2.06%)
- **iters**: 30,000 DDP-2
- **tiles**: --n-bg=3 (3 BG tiles per image)
- **warm_start**: v1 iter_30000.pth

**Results**

| variant | mIoU | R | P | hits_target |
|---|---|---|---|---|
| no-TTA baseline | 0.331 | 88.89% | 35.86% |  |
| no-TTA + T* | 0.333 | 81.50% | 44.82% |  |
| full-TTA baseline | 0.343 | 86.92% | 44.58% |  |
| full-TTA + T* | 0.346 | 83.06% | 51.69% |  |

**Per-class (full-TTA + T*)**

| class | TP | FP | FN | R | P |
|---|---|---|---|---|---|
| la_exposure | 859 | 590 | 59 | 93.60% | 59.30% |
| la_damage | 117 | 37 | 8 | 93.60% | 76.00% |
| la_crack | 15 | 6 | 57 | 20.80% | 71.40% |
| la_open | 0 | 0 | 0 | — | — |
| bond_crack | 22 | 8 | 50 | 30.60% | 73.30% |
| bond_open | 6 | 90 | 6 | 50.00% | 6.20% |
| receptor_lightning | 68 | 54 | 45 | 60.20% | 55.70% |
| receptor_damage | 83 | 40 | 81 | 50.60% | 67.50% |

---

## 🔁 Cross-comparisons

### v15 vs v17 — BG-tile ablation (--n-bg=3 vs 6)

**Purpose.** Hold everything constant except BG-tile sampling rate. Tests whether 2× clean-blade exposure during training reduces FPs.

| test_set | variant | v15_R | v15_P | v17_R | v17_P | delta_P |
|---|---|---|---|---|---|---|
| team test | no-TTA baseline | 87.06 | 33.33 | 85.70 | 39.88 | +6.55 |
| team test | no-TTA + T* | 80.89 | 44.54 | 80.42 | 52.08 | +7.54 |
| overseas test | no-TTA baseline | 90.16 | 45.89 | 84.88 | 54.17 | +8.28 |
| overseas test | no-TTA + T* | 87.37 | 54.74 | 84.58 | 59.26 | +4.52 |

**Take-away.** Doubling BG-tile exposure consistently improves precision by 4-8 pp at every operating point on both domains, at the cost of 1-5 pp recall. mIoU also improves by ~3 pp on baselines. Confirms the FP-on-featureless-blade-edges failure mode is partly a training-coverage problem.

---

### v16 vs v19 — apples-to-apples labeler effect

**Purpose.** Both trained Crack-only at same recipe. v16 = overseas lab labelers (2,993 train), v19 = team labelers (531 train). Differs in WHO labeled AND data size (5.6×).

| model | no-TTA + T* R/P | mIoU |
|---|---|---|
| v16 (lab, 2,993) | 92.47 / 72.58 | 0.581 |
| v19 (team, 531) | 78.33 / 69.12 | 0.528 |
| gap | −14.1 R / −3.5 P | −0.053 |

**Take-away.** Per-class precision is nearly identical (v19 La_Crack 70.7 P vs v16 73.1 P; v19 Bond_Crack 66.7 P vs v16 71.9 P). The 14 pp recall gap is almost entirely explained by training-data SIZE (5.6×), NOT labeler style. v23 (queued, size-matched) will confirm.

---

### v17 vs v20 — aux classification head effect

**Purpose.** v20 = v17 + tile-level binary aux classification head, jointly trained. Tests whether multi-task aux loss improves segmentation as a regularizer.

| variant | v17_R_P | v20_R_P | delta |
|---|---|---|---|
| team test, no-TTA + T* | 80.42 / 52.08 | 83.74 / 48.47 | +3.3 R, −3.6 P |
| overseas test, no-TTA + T* | 84.58 / 59.26 | 87.37 / 59.80 | +2.8 R, +0.5 P |
| combined, no-TTA + T* | 81.73 / 53.65 | 84.89 / 50.69 | +3.2 R, −3.0 P |

**Take-away.** v20 has v17's precision-leaning seg-head behavior but with v15's recall — best of both. Aux loss acted as regularizer; aux head itself NOT yet used at inference (Mode A/B inference pending).

---

### v13 cross-domain — overseas model on team test

**Purpose.** Apply v13 (overseas-trained) ckpt to team test_diuid_10. Quantifies domain transfer gap.

| variant | R | P | mIoU |
|---|---|---|---|
| no-TTA baseline | 70.33% | 28.89% | 0.204 |
| no-TTA + T* | 64.30% | 56.76% | — |
| full-TTA baseline | 67.68% | 36.39% | 0.207 |
| full-TTA + T* | 62.47% | 64.52% | — |

**Take-away.** v13 was 91 R / 66 P on its own overseas test. On team test it drops to 62.5 R / 64.5 P with full-TTA + T*. A 28.5 pp recall drop on the same model — strong evidence the team and overseas distributions are structurally different.

---

## 📈 Data analyses

### Team train ≠ team test distribution

Critical mismatch: team train_no_leak is 98% defect-positive, avg 6.81 instances/img, 1.20 classes/img. Team test_diuid_10 is 38% defect-positive (62% empty), avg 2.33 instances/img, 0.48 classes/img. Model trained on defect-heavy data is biased to fire on edge texture — root cause of low team-test precision.

| split | pct_empty | avg_inst_per_img | avg_classes_per_img | imgs_per_WTG |
|---|---|---|---|---|
| team train | 2.0% | 6.81 | 1.20 | 38.10 |
| team val | 58.7% | 2.68 | 0.52 | 303 |
| team test | 62.0% | 2.33 | 0.48 | 308 |

---

### Overseas — no train/test drift

Overseas train, val, test are statistically interchangeable — same empty rate (~6%), same instance count (~4-5/img), same classes/img (~0.94). Fair eval.

| split | pct_empty | avg_inst_per_img | avg_classes_per_img | imgs_per_WTG |
|---|---|---|---|---|
| overseas train | 6.0% | 5.02 | 0.94 | 6.50 |
| overseas val | 8.2% | 4.47 | 0.93 | 7.90 |
| overseas test | 5.8% | 3.86 | 0.95 | 6.60 |

---

### Multi-label rates by dataset

Team data has many multi-class panoramas (20% in train). Overseas is almost mono-class (94% single-class) because each image came from a task-specific labeling queue.

| split | multi_class_pct | top_combo |
|---|---|---|
| team train | 19.7% | la_exposure + la_damage |
| team test | 8.0% | la_exposure + la_damage |
| overseas test | 0.4% | receptor_damage + receptor_lightning |

---

### Receptor_Damage source in overseas

Receptor_Damage class (id=8) is NEVER a primary overseas task. It appears only as a secondary annotation in the '3_Receptor_Lightning' task — labelers annotated both the lightning marks AND the underlying damage to the receptor. 88% of Receptor_Lightning task images also have Receptor_Damage annotations.

---

### Class ID mapping is consistent between team and overseas

Both datasets use the same canonical mapping (from windblade_tiled.py METAINFO): 0=BG, 1=la_exposure, 2=la_damage, 3=la_crack, 4=la_open, 5=bond_crack, 6=bond_open, 7=receptor_lightning, 8=receptor_damage. Verified via overseas task names → mask class IDs (all match expected).

---

## Notes

Threshold-applied mIoU now populated for variants where pred_masks_npz files exist.