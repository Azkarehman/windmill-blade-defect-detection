"""Standalone evaluation of an mmseg checkpoint — matches team's metrics.

For each test image:
  1. Run sliding-window inference at cfg.test_cfg (typically crop=1024, stride=512
     or SAM's crop=1008, stride=504).
  2. Accumulate per-class pixel intersection/union for IoU.
  3. Accumulate per-class image-level TP/FP/FN/TN with bbox-overlap detection
     rule (TP iff any pred-CC bbox overlaps any GT-CC bbox for that class).
     This matches mmseg/engine/hooks/fast_val_hook.py:246-272.

Optionally merges 9-class output to 5-class merged taxonomy for cross-experiment
comparison. Saves metrics, confusion matrix, and a few side-by-side overlays.

Usage:
  python eval_seg.py \\
      --config seg/configs/sam3_lora_windblade_8class.py \\
      --checkpoint runs/sam3_lora_v1/iter_30000.pth \\
      --test-json /home/work/workspace/jongwon/dmdd/data/test_diuid_17wtg.json \\
      --out-dir runs/sam3_lora_v1/eval_17wtg \\
      [--merge-to-5class]
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
# So that `import sam3_backbone` / `import windblade_merged` registers them.
import sam3_backbone  # noqa: F401
import windblade_merged  # noqa: F401

from mmseg.apis import init_model, inference_model


# 9-class → 5-class merged LUT (matches common.py / windblade_merged.py)
LUT_9TO5 = np.zeros(256, dtype=np.uint8)
LUT_9TO5[1] = 1; LUT_9TO5[2] = 1
LUT_9TO5[3] = 2; LUT_9TO5[4] = 2
LUT_9TO5[5] = 3; LUT_9TO5[6] = 3
LUT_9TO5[7] = 4; LUT_9TO5[8] = 4
LUT_9TO5[255] = 255

CLASS_NAMES_9 = [
    "Background", "La_Exposure", "La_Damage", "La_Crack", "La_Open",
    "Bond_Crack", "Bond_Open", "Receptor_Lightning", "Receptor_Damage",
]
CLASS_NAMES_5 = [
    "Background", "Laminate_Surface", "Laminate_Crack", "Bond", "Receptor",
]

# Distinct palette per class for visualizations
PALETTE_5 = np.array([[0, 0, 0], [30, 144, 255], [220, 20, 60],
                      [0, 200, 100], [255, 215, 0]], dtype=np.uint8)
PALETTE_9 = np.array([[0, 0, 0], [30, 144, 255], [0, 100, 255], [220, 20, 60],
                      [180, 0, 50], [0, 200, 100], [0, 130, 60],
                      [255, 215, 0], [200, 150, 0]], dtype=np.uint8)


def bbox_overlap(pred_cc_stats, gt_cc_stats) -> bool:
    """True iff any predicted-CC bbox intersects any GT-CC bbox."""
    for i in range(1, len(pred_cc_stats)):
        px, py, pw, ph = pred_cc_stats[i, :4]
        for j in range(1, len(gt_cc_stats)):
            gx, gy, gw, gh = gt_cc_stats[j, :4]
            if px < gx + gw and gx < px + pw and py < gy + gh and gy < py + ph:
                return True
    return False


def overlay_mask(img_rgb, mask, palette, alpha=0.55):
    out = img_rgb.copy()
    for cls in range(1, len(palette)):
        m = mask == cls
        if not m.any():
            continue
        out[m] = ((1 - alpha) * out[m] + alpha * palette[cls]).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-json", default="/home/work/workspace/jongwon/dmdd/data/test_diuid_10.json",
                    help="Default: test_diuid_10 (3,083 imgs / 10 WTGs, strictly held out from val_diuid_7 used by FastValHook).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-vis", type=int, default=20,
                    help="Number of side-by-side visualizations to save.")
    ap.add_argument("--max-images", type=int, default=None,
                    help="Eval at most N images (for quick partial eval).")
    ap.add_argument("--merge-to-5class", action="store_true",
                    help="If model outputs 9 classes, merge pred+GT to 5-class "
                         "via the standard LUT before computing metrics.")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "visualizations").mkdir(parents=True, exist_ok=True)

    entries = json.load(open(args.test_json))
    if args.max_images:
        entries = entries[: args.max_images]
        print(f"Test entries: {len(entries)} (capped via --max-images)")
    else:
        print(f"Test entries: {len(entries)}")
    print(f"Loading model from {args.checkpoint} ...")
    t0 = time.time()
    model = init_model(args.config, args.checkpoint, device=args.device)
    print(f"  loaded in {time.time()-t0:.1f}s")

    # Infer num_classes from model's decode_head
    raw_num_classes = model.decode_head.num_classes
    # If we asked for merge and model is 9-class, evaluate in 5-class space.
    if args.merge_to_5class and raw_num_classes == 9:
        eval_num_classes = 5
        class_names = CLASS_NAMES_5
        palette = PALETTE_5
        do_merge = True
    else:
        eval_num_classes = raw_num_classes
        class_names = (CLASS_NAMES_9 if raw_num_classes == 9
                       else CLASS_NAMES_5 if raw_num_classes == 5
                       else [f"class_{i}" for i in range(raw_num_classes)])
        palette = PALETTE_9 if raw_num_classes == 9 else PALETTE_5
        do_merge = False
    print(f"raw num_classes={raw_num_classes}, eval num_classes={eval_num_classes}, do_merge={do_merge}")

    # Per-class accumulators
    total_intersection = np.zeros(eval_num_classes, dtype=np.int64)
    total_union = np.zeros(eval_num_classes, dtype=np.int64)
    total_pred_pix = np.zeros(eval_num_classes, dtype=np.int64)
    total_gt_pix = np.zeros(eval_num_classes, dtype=np.int64)
    img_det = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0})

    n_processed = 0
    n_skipped = 0
    t0 = time.time()
    for idx, entry in enumerate(tqdm(entries, desc="eval")):
        img_path = entry["image_path"]
        gt_path = entry["mask_path"]
        if not (Path(img_path).exists() and Path(gt_path).exists()):
            n_skipped += 1
            continue

        try:
            result = inference_model(model, img_path)
        except Exception as e:
            print(f"  inference fail {Path(img_path).name}: {e}")
            n_skipped += 1
            continue
        pred = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if gt is None:
            n_skipped += 1
            continue

        # Resize pred to GT size if they differ
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        # Apply 9→5 merge if requested
        if do_merge:
            pred = LUT_9TO5[pred]
            gt = LUT_9TO5[gt]

        # Pixel-level intersection/union per class
        valid = gt != 255  # ignore_index
        for cls in range(eval_num_classes):
            pm = (pred == cls) & valid
            gm = (gt == cls) & valid
            inter = int((pm & gm).sum())
            union = int((pm | gm).sum())
            total_intersection[cls] += inter
            total_union[cls] += union
            total_pred_pix[cls] += int(pm.sum())
            total_gt_pix[cls] += int(gm.sum())

        # Image-level bbox-overlap TP/FP/FN/TN (foreground classes only)
        for cls in range(1, eval_num_classes):
            gt_cls = ((gt == cls) & valid).astype(np.uint8)
            pred_cls = ((pred == cls) & valid).astype(np.uint8)
            gt_has = bool(gt_cls.any())
            pred_has = bool(pred_cls.any())
            overlap = False
            if gt_has and pred_has:
                _, _, gt_stats, _ = cv2.connectedComponentsWithStats(gt_cls, connectivity=8)
                _, _, pred_stats, _ = cv2.connectedComponentsWithStats(pred_cls, connectivity=8)
                overlap = bbox_overlap(pred_stats, gt_stats)
            if overlap:
                img_det[cls]["TP"] += 1
            else:
                if pred_has:
                    img_det[cls]["FP"] += 1
                if gt_has:
                    img_det[cls]["FN"] += 1
                if not gt_has and not pred_has:
                    img_det[cls]["TN"] += 1

        # Save a few side-by-side overlays
        if n_processed < args.num_vis:
            img_bgr = cv2.imread(img_path)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                # Downsize for the overlay so the JPG isn't huge
                H_target = 700
                s = H_target / img_rgb.shape[0]
                W = int(img_rgb.shape[1] * s)
                img_s = cv2.resize(img_rgb, (W, H_target))
                gt_s = cv2.resize(gt, (W, H_target), interpolation=cv2.INTER_NEAREST)
                pred_s = cv2.resize(pred, (W, H_target), interpolation=cv2.INTER_NEAREST)
                gt_ov = overlay_mask(img_s, gt_s, palette)
                pred_ov = overlay_mask(img_s, pred_s, palette)
                grid = np.hstack([img_s, gt_ov, pred_ov])
                cv2.putText(grid, "source", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(grid, "GT", (img_s.shape[1] + 10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(grid, "Pred", (2 * img_s.shape[1] + 10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                vis_path = out / "visualizations" / f"{idx:04d}_{Path(img_path).stem}.jpg"
                cv2.imwrite(str(vis_path),
                            cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])

        n_processed += 1

    elapsed = time.time() - t0
    print(f"\nProcessed {n_processed} (skipped {n_skipped}) in {elapsed/60:.1f} min "
          f"({elapsed/max(1, n_processed):.2f} s/img)")

    # ── Compute per-class metrics
    per_class = []
    valid_ious = []
    valid_recalls = []
    valid_precisions = []
    for cls in range(eval_num_classes):
        iou = (total_intersection[cls] / total_union[cls]
               if total_union[cls] > 0 else float("nan"))
        dice = (2 * total_intersection[cls] /
                (total_pred_pix[cls] + total_gt_pix[cls])
                if (total_pred_pix[cls] + total_gt_pix[cls]) > 0 else float("nan"))
        if cls > 0:
            s = img_det[cls]
            tp, fp, fn = s["TP"], s["FP"], s["FN"]
            recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 and not np.isnan(precision)
                  and not np.isnan(recall) else float("nan"))
            if not np.isnan(recall):
                valid_recalls.append(recall)
            if not np.isnan(precision):
                valid_precisions.append(precision)
        else:
            tp = fp = fn = 0
            recall = precision = f1 = float("nan")
        if not np.isnan(iou):
            valid_ious.append(iou)
        per_class.append({
            "class": class_names[cls],
            "iou": iou,
            "dice": dice,
            "pixel_gt": int(total_gt_pix[cls]),
            "pixel_pred": int(total_pred_pix[cls]),
            "img_tp": int(tp),
            "img_fp": int(fp),
            "img_fn": int(fn),
            "img_recall": recall,
            "img_precision": precision,
            "img_f1": f1,
        })

    miou = float(np.nanmean([c["iou"] for c in per_class[1:]]))    # fg only
    miou_all = float(np.nanmean([c["iou"] for c in per_class]))    # incl BG
    mean_recall = float(np.mean(valid_recalls)) if valid_recalls else float("nan")
    mean_precision = float(np.mean(valid_precisions)) if valid_precisions else float("nan")

    print(f"\n=== Pixel-level (per-class IoU) ===")
    print(f"{'Class':<22} {'IoU':>8} {'Dice':>8}")
    print("-" * 42)
    for r in per_class:
        iou_s = f"{r['iou']*100:>7.2f}%" if not np.isnan(r["iou"]) else "    N/A"
        dice_s = f"{r['dice']*100:>7.2f}%" if not np.isnan(r["dice"]) else "    N/A"
        print(f"{r['class']:<22} {iou_s} {dice_s}")
    print("-" * 42)
    print(f"{'mIoU (fg only)':<22} {miou*100:>7.2f}%")
    print(f"{'mIoU (incl BG)':<22} {miou_all*100:>7.2f}%")

    print(f"\n=== Image-level (bbox-overlap TP/FP/FN) ===")
    print(f"{'Class':<22} {'TP':>5} {'FP':>5} {'FN':>5} {'Recall':>9} {'Prec':>9} {'F1':>9}")
    print("-" * 70)
    for r in per_class[1:]:
        rc = f"{r['img_recall']:.3f}" if not np.isnan(r["img_recall"]) else "N/A"
        pr = f"{r['img_precision']:.3f}" if not np.isnan(r["img_precision"]) else "N/A"
        f1 = f"{r['img_f1']:.3f}" if not np.isnan(r["img_f1"]) else "N/A"
        print(f"{r['class']:<22} {r['img_tp']:>5} {r['img_fp']:>5} {r['img_fn']:>5} "
              f"{rc:>9} {pr:>9} {f1:>9}")
    print("-" * 70)
    print(f"{'Mean (fg)':<22} {'':>5} {'':>5} {'':>5} "
          f"{mean_recall:>9.3f} {mean_precision:>9.3f}")

    # ── Save
    metrics = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "test_json": args.test_json,
        "n_processed": n_processed,
        "n_skipped": n_skipped,
        "raw_num_classes": int(raw_num_classes),
        "eval_num_classes": int(eval_num_classes),
        "merged_to_5class": do_merge,
        "miou_fg": miou,
        "miou_all": miou_all,
        "mean_recall": mean_recall,
        "mean_precision": mean_precision,
        "per_class": per_class,
    }
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2,
              default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)

    # Confusion matrix at pixel level
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(0.9 * eval_num_classes + 2, 0.9 * eval_num_classes + 2))
    cm_norm = np.zeros((eval_num_classes, eval_num_classes))
    # Approximate confusion: per-class precision row + recall col (we don't have
    # the full GT-vs-pred contingency without re-iteration). Use only diagonal
    # = TP_pix / (TP+FN)_pix per class instead.
    for c in range(eval_num_classes):
        if total_gt_pix[c] > 0:
            cm_norm[c, c] = total_intersection[c] / total_gt_pix[c]
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(eval_num_classes)); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(eval_num_classes)); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("GT")
    ax.set_title(f"{Path(args.checkpoint).stem}\nmIoU(fg) {miou*100:.2f}%  "
                 f"meanR {mean_recall:.3f}  meanP {mean_precision:.3f}")
    for i in range(eval_num_classes):
        ax.text(i, i, f"{cm_norm[i, i]*100:.0f}%", ha="center", va="center",
                color="white" if cm_norm[i, i] > 0.5 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out / "per_class_recall.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote {out}/metrics.json, {out}/per_class_recall.png, {out}/visualizations/*")


if __name__ == "__main__":
    main()
