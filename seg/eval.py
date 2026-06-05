"""
Evaluate a trained semseg model on a val/test JSON.

Metrics:
  - per-class IoU + mIoU
  - per-class Dice + mean Dice
  - Precision, Recall (pixel-level, per class)
  - confusion matrix (saved as PNG + JSON)

Works for both binary and 5-class merged. Uses sliding-window inference
matching the model's test_cfg (crop 1024, stride 512). Applies the merged LUT
to GT masks on load.

Usage:
    python seg/eval.py \\
        --config seg/configs/segformer_b4_merged5.py \\
        --checkpoint <work_dir>/best_mIoU_iter_XXXX.pth \\
        --data-json /home/work/workspace/jongwon/dmdd/data/test_diuid_10.json \\
        --out runs/seg_eval_v1 \\
        --merge-mode 5class
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import LUT, MERGED_NAMES_FULL  # 5-class merged names + LUT

# Binary names + LUT
BINARY_NAMES = ["background", "defect"]
LUT_BIN = np.zeros(256, dtype=np.uint8)
LUT_BIN[1:9] = 1
LUT_BIN[255] = 255


def relabel(mask_9class: np.ndarray, mode: str) -> np.ndarray:
    if mode == "binary":
        return LUT_BIN[mask_9class]
    return LUT[mask_9class]


def plot_confusion(cm: np.ndarray, names, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(0.9 * len(names) + 2, 0.9 * len(names) + 2))
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.where(row_sum > 0, cm / np.maximum(row_sum, 1), 0)
    im = ax.imshow(norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted (pixel)"); ax.set_ylabel("Ground truth (pixel)")
    ax.set_title(title)
    for i in range(len(names)):
        for j in range(len(names)):
            txt = f"{cm[i, j]}\n{norm[i, j]*100:.1f}%"
            color = "white" if norm[i, j] > 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="mmseg config used for training")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-json", required=True,
                    help="JSON list of {image_path, mask_path} to evaluate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge-mode", choices=["5class", "binary"], default="5class")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    # PyTorch 2.6 compat
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})

    from mmseg.apis import init_model
    model = init_model(args.config, args.checkpoint, device=args.device)

    names = MERGED_NAMES_FULL if args.merge_mode == "5class" else BINARY_NAMES
    n_classes = len(names)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = json.load(open(args.data_json))
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    skipped = 0

    from mmseg.structures import SegDataSample
    for entry in tqdm(entries, desc="seg-eval"):
        img_path = entry["image_path"]
        msk_path = entry["mask_path"]
        img = cv2.imread(img_path)
        gt9 = np.array(Image.open(msk_path))
        if img is None or gt9 is None:
            skipped += 1
            continue
        gt = relabel(gt9, args.merge_mode)

        with torch.no_grad():
            img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(args.device)
            ds = SegDataSample()
            ds.set_metainfo({"img_shape": img.shape[:2],
                             "ori_shape": img.shape[:2],
                             "pad_shape": img.shape[:2],
                             "scale_factor": (1.0, 1.0)})
            data = {"inputs": img_t, "data_samples": [ds]}
            data = model.data_preprocessor(data, False)
            result = model.predict(data["inputs"], data["data_samples"])[0]
            pred = result.pred_sem_seg.data.cpu().numpy()[0]

        # Pixel-wise confusion (ignore 255)
        valid = gt != 255
        gt_v = gt[valid].astype(np.int64)
        pr_v = pred[valid].astype(np.int64)
        idx = gt_v * n_classes + pr_v
        binc = np.bincount(idx, minlength=n_classes * n_classes)
        cm += binc.reshape(n_classes, n_classes)

    # ----- per-class IoU + Dice + P/R -----
    per_class = []
    for i, name in enumerate(names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom_iou = tp + fp + fn
        denom_dice = 2 * tp + fp + fn
        iou = tp / denom_iou if denom_iou > 0 else float("nan")
        dice = 2 * tp / denom_dice if denom_dice > 0 else float("nan")
        p = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        r = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        per_class.append({"class": name, "IoU": float(iou), "Dice": float(dice),
                          "precision": float(p), "recall": float(r),
                          "support_pixels": int(cm[i, :].sum())})
    miou = float(np.nanmean([c["IoU"] for c in per_class]))
    mdice = float(np.nanmean([c["Dice"] for c in per_class]))
    pixel_acc = float(np.diag(cm).sum() / max(1, cm.sum()))

    # ----- printed summary -----
    print(f"\n=== SEG METRICS ({args.merge_mode}) ===")
    print(f"Pixel accuracy:   {pixel_acc:.4f}")
    print(f"mIoU:             {miou:.4f}")
    print(f"mean Dice:        {mdice:.4f}\n")
    print(f"{'Class':<22} {'IoU':>8} {'Dice':>8} {'P':>8} {'R':>8} {'support':>12}")
    print("-" * 72)
    for r in per_class:
        f = lambda x: f"{x:.4f}" if isinstance(x, float) and not np.isnan(x) else "N/A"
        print(f"{r['class']:<22} {f(r['IoU']):>8} {f(r['Dice']):>8} "
              f"{f(r['precision']):>8} {f(r['recall']):>8} "
              f"{r['support_pixels']:>12d}")

    print(f"\nConfusion matrix (pixel counts; rows=GT, cols=Pred):")
    print(" " * 24 + " ".join(f"{n[:12]:>12}" for n in names))
    for i, name in enumerate(names):
        print(f"{name:<22}  " + " ".join(f"{cm[i,j]:>12d}" for j in range(n_classes)))

    plot_confusion(cm, names, out_dir / "confusion_matrix.png",
                   f"Seg {args.merge_mode} pixel confusion (mIoU={miou:.3f})")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"merge_mode": args.merge_mode,
                   "pixel_accuracy": pixel_acc,
                   "mIoU": miou, "mean_Dice": mdice,
                   "per_class": per_class,
                   "confusion_matrix": cm.tolist(),
                   "skipped": skipped}, f, indent=2)
    print(f"\nSaved: {out_dir}/confusion_matrix.png, {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
