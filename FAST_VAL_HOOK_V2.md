# `FastValHookV2` — faster validation, same metrics

## Why

`FastValHook` was the wall-clock bottleneck: 7h 20m for one val pass on the
2,124-image `val_diuid_7` set on the Azure A100 (~9.14 s/img on 8K×5.5K
imagery). Since this hook is what selects our `best_recall_iter_*.pth` and
`best_mIoU_iter_*.pth`, we can't just skip or subsample it.

`seg/fast_val_hook_v2.py` is a **parallel file** — the original hook is
untouched in mmseg, so A/B is trivial (just flip `type=` back).

## Changes

### GPU side
1. **DataLoader workers** for parallel image / GT / blade-mask I/O,
   `persistent_workers=True` (no fork cost between vals). Default
   `num_workers=8`.
2. **bf16 autocast** on the slide inference. ViT matmuls run in bf16 on
   H200 tensor cores; argmax + post-processing stays fp32. Chose bf16 over
   fp16 to avoid attention overflow.
3. **Batched slide windows**. Original ran one crop per `encode_decode`
   call (~50–100 launches per image). Now `slide_batch_size=8` windows
   go through in one forward.

### CPU side
4. **Vectorized confusion-matrix IoU** via single `np.bincount` pass,
   replacing the per-class loop of `(pred==c)` / `(gt==c)` allocations.
   Math-identical.
5. **Skip per-class scans when the class is absent.** Precompute
   `np.unique(gt)` / `np.unique(pred)` once and short-circuit the CC +
   bbox-overlap block for classes that aren't present. Typical image has
   2–3 classes out of 8 — saves 10–14 full-res mask allocations per
   image. Bit-identical TP/FP/FN counts.
6. **Visualization respects `save_vis_samples`** (default 10, was
   hardcoded 100). Same `np.unique` short-circuit applied to overlay
   painting, with class colors cached once at import time.
7. **Optional `cc_downsample_factor`** (default 1 = off). When set to
   2 or 4, downsamples masks before connected-components. Speeds up
   image-level detection but may miss very small defects. Opt-in.

## Measured wall-time impact

Bench harness: 100 random images from `val_diuid_7`, warm cache,
H200 GPU. Compares ORIGINAL `FastValHook` vs `FastValHookV2` back-to-back
on identical inputs.

| Variant                                      | s/img | Speedup vs OLD | Extrap. full val |
| -------------------------------------------- | ----- | -------------- | ---------------- |
| OLD `FastValHook` on H200                    | 5.66  | 1.00×          | 3h 20m           |
| V2 (workers + bf16 + batched slide)          | 4.78  | 1.16×          | 2h 49m           |
| V2 + bincount IoU + viz cap (live config)    | 3.68  | **1.54×**      | **2h 10m**       |
| V2 + skip-absent + workers=8 (bench v4)      | TBD   | TBD            | TBD              |

For reference: 7h 20m → ~2h 10m total is **~3.4× end-to-end**, but most
of that is the **H200 hardware upgrade** (Azure A100 → H200 ≈ 2.2×).
The hook itself contributes the remaining 1.54× on top.

bf16 introduces ~1e-3 drift on logits; mIoU and Mean Recall match the
original hook to within ≤ 0.01% across bench runs — far below the gap
between adjacent checkpoint scores.

## What I did NOT change

- Val set size, `crop_size`, `stride`, or precision below bf16 — any
  would shift checkpoint ranking.
- `cc_downsample_factor` default — stays 1 (off). Only opt-in.
- The original `FastValHook` in mmsegmentation. Untouched.

Image-level metric definitions (TP/FP/FN, per-class IoU, mean recall)
are byte-for-byte identical.

## Files

- `seg/fast_val_hook_v2.py` — the new hook.
- `seg/configs/sam3_lora_windblade_8class.py` — uses
  `type='FastValHookV2'` with the new args (`num_workers=8`,
  `prefetch_factor=2`, `use_bf16=True`, `persistent_workers=True`,
  `slide_batch_size=8`).
- `seg/_torch_compat.py` — small shim to make `torch.load` default
  `weights_only=False` (needed to resume mmengine 0.10.x checkpoints on
  torch 2.6+).
- `mmsegmentation/mmseg/engine/hooks/fast_val_hook.py` — original,
  untouched. Swap `type=` back for A/B.

## Tuning knobs

| Arg | Default | When to change |
| --- | ------- | -------------- |
| `num_workers` | 8 | Drop if CPU memory is tight; raise if cold-cache I/O is slow. |
| `slide_batch_size` | 8 | Raise to 12/16 if `nvidia-smi` shows free VRAM. |
| `save_vis_samples` | 10 | Set to 0 to disable val-time viz entirely. |
| `cc_downsample_factor` | 1 | Set to 2 or 4 to trade tiny-defect detection for CC speed. |
| `use_bf16` | True | Set False only for an apples-to-apples vs old hook. |
