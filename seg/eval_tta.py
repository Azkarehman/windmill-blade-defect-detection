"""TTA evaluation for SAM 3.1 LoRA on test_diuid_10, parallel + multi-GPU.

Output format aligned with the team's
``/home/work/workspace/jongwon/dmdd/scripts/eval_with_records.py`` — same
per-image JSONL record schema and summary JSON schema, so downstream
analysis tooling that reads their files reads ours too.

Smart design:
  - Runs 6 augmentation views per image in ONE inference pass.
  - Snapshots the accumulated logits at 3 view counts:
        view_count=1 (baseline, no TTA)
        view_count=3 (half TTA = orig + h-flip + scale 1.25)
        view_count=6 (full TTA = adds scale 1.25 flip + scale 0.75 ±flip)

Parallelism:
  - ``--num-workers N``  : DataLoader workers prefetch (image, gt, blade) on
    CPU while the GPU is busy with inference.
  - ``--shard-id i --num-shards N`` : process only every N-th test image
    starting at offset i. Launch two of these processes (one per GPU) for
    2x wall-clock reduction. The wrapper ``run_tta_weekend.sh`` does that.
  - ``--merge`` : combine per-shard JSONLs + summaries into final unified
    files. Run after both shard processes finish.

Per-shard outputs (under <output_dir>/, suffixed when --num-shards > 1):
  <level>_records.shard{i}of{N}.jsonl       — per-image rows
  <level>_summary.shard{i}of{N}.json        — aggregate metrics for this shard
                                              (incl. raw inter/union for merge)
After --merge:
  <level>_records.jsonl, <level>_summary.json — combined final outputs.

Usage (parallel 2-GPU recipe is in run_tta_weekend.sh; single-GPU below):
    python eval_tta.py \
        --config <config>.py \
        --checkpoint <ckpt>.pth \
        --test-json <test.json> \
        --blade-filter-mask-dir <blade>/ \
        --output-dir <out>/ \
        --num-workers 4
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
from tqdm import tqdm

# Make our seg/ modules importable
sys.path.insert(0, '/home/work/workspace/azka5/dmdd_pipeline/seg')
import _torch_compat   # noqa: F401
import sam3_backbone   # noqa: F401

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TTA_VIEWS = [
    {'scale': 1.00, 'flip': False, 'label': 'orig'},
    {'scale': 1.00, 'flip': True,  'label': 'orig+flip'},
    {'scale': 1.25, 'flip': False, 'label': 'scale1.25'},
    # ↑ HALF TTA snapshot (3 views averaged)
    {'scale': 1.25, 'flip': True,  'label': 'scale1.25+flip'},
    {'scale': 0.75, 'flip': False, 'label': 'scale0.75'},
    {'scale': 0.75, 'flip': True,  'label': 'scale0.75+flip'},
    # ↑ FULL TTA snapshot (6 views averaged)
]
SNAPSHOT_LEVELS = [
    (1, 'baseline'),
    (3, 'half_tta'),
    (6, 'full_tta'),
]
SNAPSHOT_VIEW_SET = {v for v, _ in SNAPSHOT_LEVELS}

CLASS_NAMES = [
    'Background', 'La Exposure', 'La Damage', 'La Crack',
    'La Open', 'Bond Crack', 'Bond Open', 'Receptor Lightning',
    'Receptor Damage',
]
NUM_CLASSES = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Team-compatible helpers (mirror scripts/eval_with_records.py)
# ---------------------------------------------------------------------------

def cc_stats(binary_mask):
    if not binary_mask.any():
        return []
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, len(stats)):
        x, y, w, h, _ = stats[i]
        out.append([int(y), int(x), int(h), int(w)])
    return out


def per_class_info(mask):
    out = {}
    for c in range(1, NUM_CLASSES):
        bm = (mask == c)
        if not bm.any():
            continue
        bboxes = cc_stats(bm)
        out[str(c)] = {
            'pixels': int(bm.sum()),
            'n_components': len(bboxes),
            'bboxes': bboxes,
        }
    return out


def instance_overlap(gt_mask, pred_mask, cls_id):
    gt_cls = (gt_mask == cls_id).astype(np.uint8)
    pred_cls = (pred_mask == cls_id).astype(np.uint8)
    gt_has = bool(gt_cls.any())
    pred_has = bool(pred_cls.any())
    overlap = False
    if gt_has and pred_has:
        _, _, gs, _ = cv2.connectedComponentsWithStats(gt_cls, connectivity=8)
        _, _, ps, _ = cv2.connectedComponentsWithStats(pred_cls, connectivity=8)
        for i in range(1, len(ps)):
            px, py, pw, ph = ps[i, :4]
            for j in range(1, len(gs)):
                gx, gy, gw, gh = gs[j, :4]
                if (px < gx + gw and gx < px + pw
                        and py < gy + gh and gy < py + ph):
                    overlap = True
                    break
            if overlap:
                break
    return {
        'gt_has': gt_has, 'pred_has': pred_has, 'overlap': overlap,
        'TP': int(overlap),
        'FP': int(pred_has and not overlap),
        'FN': int(gt_has and not overlap),
    }


def per_image_iou(gt, pred, num_classes=NUM_CLASSES):
    out = {}
    for c in range(num_classes):
        gm = (gt == c)
        pm = (pred == c)
        inter = int(np.logical_and(gm, pm).sum())
        union = int(np.logical_or(gm, pm).sum())
        out[str(c)] = (inter / union) if union > 0 else float('nan')
    return out


def load_blade_mask(img_path, target_hw, mask_dir, classes=(1, 2)):
    wtg = Path(img_path).parent.name
    stem = Path(img_path).stem
    mask_path = Path(mask_dir) / wtg / f'{stem}.png'
    if not mask_path.exists():
        return None
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    blade = np.isin(m, classes).astype(np.uint8)
    h, w = target_hw
    if blade.shape != (h, w):
        blade = cv2.resize(blade, (w, h), interpolation=cv2.INTER_NEAREST)
    return blade


# ---------------------------------------------------------------------------
# Dataset for parallel CPU-side loading (multi-worker prefetch)
# ---------------------------------------------------------------------------

class TestImageDataset(torch.utils.data.Dataset):
    """Loads (image, gt_mask, blade_mask, metadata) in DataLoader workers.

    Returns None for images with missing files; the collate passes None
    through so the main loop can skip them.
    """

    def __init__(self, samples, blade_mask_dir):
        self.samples = samples
        self.blade_mask_dir = blade_mask_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample['image_path']
        mask_path = sample['mask_path']
        if not (os.path.exists(image_path) and os.path.exists(mask_path)):
            return None
        image = cv2.imread(image_path)
        if image is None:
            return None
        gt_mask = np.array(Image.open(mask_path))
        blade_mask = None
        blade_pixel_ratio = None
        if self.blade_mask_dir is not None:
            blade_mask = load_blade_mask(image_path, image.shape[:2],
                                          self.blade_mask_dir)
            if blade_mask is not None:
                blade_pixel_ratio = float(blade_mask.mean())
        return {
            'image': image,
            'gt_mask': gt_mask,
            'blade_mask': blade_mask,
            'blade_pixel_ratio': blade_pixel_ratio,
            'image_path': image_path,
            'wtg': Path(image_path).parent.name,
            'stem': Path(image_path).stem,
        }


def _passthrough_collate(batch):
    return batch[0]


# ---------------------------------------------------------------------------
# Slide inference returning raw logits
# ---------------------------------------------------------------------------

def slide_logits(model, inputs, base_meta, blade_mask_np=None):
    test_cfg = model.test_cfg
    h_stride, w_stride = test_cfg['stride']
    h_crop, w_crop = test_cfg['crop_size']
    bs, _, h_img, w_img = inputs.shape
    out_channels = model.decode_head.out_channels

    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    preds = inputs.new_zeros((bs, out_channels, h_img, w_img))
    count = inputs.new_zeros((bs, 1, h_img, w_img))

    crop_meta = dict(base_meta)
    crop_meta['img_shape'] = torch.Size([h_crop, w_crop])

    coords = []
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            if blade_mask_np is not None and blade_mask_np[y1:y2, x1:x2].sum() == 0:
                continue
            coords.append((y1, x1, y2, x2))

    if not coords:
        return preds

    SLIDE_BATCH = 8
    for i in range(0, len(coords), SLIDE_BATCH):
        chunk = coords[i:i + SLIDE_BATCH]
        crops = torch.cat(
            [inputs[:, :, y1:y2, x1:x2] for (y1, x1, y2, x2) in chunk], dim=0)
        metas = [dict(crop_meta) for _ in chunk]
        chunk_logits = model.encode_decode(crops, metas)
        for j, (y1, x1, y2, x2) in enumerate(chunk):
            preds[:, :, y1:y2, x1:x2] += chunk_logits[j:j + 1]
            count[:, :, y1:y2, x1:x2] += 1
    count = count.clamp(min=1)
    return preds / count


# ---------------------------------------------------------------------------
# Per-view inference
# ---------------------------------------------------------------------------

def run_one_view(model, data_preprocessor, image_t_orig, blade_mask_orig,
                  scale, flip, orig_hw, device):
    H, W = orig_hw
    img = image_t_orig

    if scale != 1.0:
        new_h = int(round(H * scale))
        new_w = int(round(W * scale))
        img = F.interpolate(img, size=(new_h, new_w),
                            mode='bilinear', align_corners=False)
        if blade_mask_orig is not None:
            bm = torch.from_numpy(blade_mask_orig).float().unsqueeze(0).unsqueeze(0)
            bm = F.interpolate(bm, size=(new_h, new_w), mode='nearest')
            blade_mask_view = bm[0, 0].byte().numpy()
        else:
            blade_mask_view = None
    else:
        blade_mask_view = blade_mask_orig

    if flip:
        img = torch.flip(img, dims=[-1])
        if blade_mask_view is not None:
            blade_mask_view = blade_mask_view[:, ::-1].copy()

    view_h, view_w = img.shape[-2:]
    meta = {
        'img_shape': (view_h, view_w),
        'ori_shape': (view_h, view_w),
        'pad_shape': (view_h, view_w),
        'scale_factor': (1.0, 1.0),
    }
    data_sample = SegDataSample()
    data_sample.set_metainfo(meta)
    data = {'inputs': img, 'data_samples': [data_sample]}
    data = data_preprocessor(data, False)

    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        logits = slide_logits(model, data['inputs'], meta, blade_mask_view)

    if flip:
        logits = torch.flip(logits, dims=[-1])
    if scale != 1.0:
        logits = F.interpolate(logits.float(), size=(H, W),
                               mode='bilinear', align_corners=False)
    return logits.float()


# ---------------------------------------------------------------------------
# Per-image processing (takes pre-loaded sample from DataLoader worker)
# ---------------------------------------------------------------------------

def process_image(model, data_preprocessor, prepared, args, output_state):
    image = prepared['image']
    gt_mask = prepared['gt_mask']
    blade_mask = prepared['blade_mask']
    blade_pixel_ratio = prepared['blade_pixel_ratio']
    image_path = prepared['image_path']
    wtg = prepared['wtg']
    stem = prepared['stem']
    H, W = image.shape[:2]

    image_t = torch.from_numpy(image).permute(2, 0, 1).float()
    image_t = image_t.unsqueeze(0).to(args.device, non_blocking=True)

    accum = None
    t_inf0 = time.time()

    for view_idx, view in enumerate(TTA_VIEWS):
        logits = run_one_view(model, data_preprocessor, image_t, blade_mask,
                              view['scale'], view['flip'], (H, W), args.device)
        accum = logits if accum is None else accum + logits

        view_count = view_idx + 1
        if view_count not in SNAPSHOT_VIEW_SET:
            del logits
            continue

        avg = accum / view_count
        pred = avg.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        if blade_mask is not None:
            pred = pred.copy()
            pred[blade_mask == 0] = 0
        inf_time_so_far = time.time() - t_inf0

        inst_det = {}
        for c in range(1, NUM_CLASSES):
            inst_det[str(c)] = instance_overlap(gt_mask, pred, c)
        iou_img = per_image_iou(gt_mask, pred)

        level_name = next(name for vc, name in SNAPSHOT_LEVELS if vc == view_count)
        record = {
            'image_path': image_path,
            'wtg': wtg,
            'stem': stem,
            'image_shape': [H, W],
            'blade_pixel_ratio': blade_pixel_ratio,
            'inference_time': inf_time_so_far,
            'tta_views': view_count,
            'gt': per_class_info(gt_mask),
            'pred': per_class_info(pred),
            'instance_detection': inst_det,
            'iou': iou_img,
        }
        output_state['records_jf'][level_name].write(json.dumps(record) + '\n')
        output_state['records_jf'][level_name].flush()

        agg = output_state['agg'][level_name]
        for c in range(NUM_CLASSES):
            gm = (gt_mask == c)
            pm = (pred == c)
            agg['total_inter'][c] += int(np.logical_and(gm, pm).sum())
            agg['total_union'][c] += int(np.logical_or(gm, pm).sum())
        for c in range(1, NUM_CLASSES):
            d = inst_det[str(c)]
            agg['img_det'][c]['TP'] += d['TP']
            agg['img_det'][c]['FP'] += d['FP']
            agg['img_det'][c]['FN'] += d['FN']
            if not d['gt_has'] and not d['pred_has']:
                agg['img_det'][c]['TN'] += 1
        agg['n_samples'] += 1
        agg['total_time'] += inf_time_so_far / len(SNAPSHOT_LEVELS)

        if args.save_pred_masks:
            pred_dir = output_state['pred_dirs'][level_name] / wtg
            pred_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(pred_dir / f'{stem}.png'), pred)

        del logits

    del image_t, accum
    torch.cuda.empty_cache()
    return image_path


# ---------------------------------------------------------------------------
# Summary I/O (per-shard + final merged)
# ---------------------------------------------------------------------------

def build_summary_dict(agg):
    per_class_iou_global = {}
    valid_ious = []
    for c in range(NUM_CLASSES):
        if agg['total_union'][c] > 0:
            iou = agg['total_inter'][c] / agg['total_union'][c]
            per_class_iou_global[c] = iou
            valid_ious.append(iou)
        else:
            per_class_iou_global[c] = float('nan')
    miou = float(np.nanmean(valid_ious)) if valid_ious else 0.0

    img_det_results = {}
    valid_r, valid_p = [], []
    total_tp = 0; total_fp = 0; total_fn = 0
    for c in range(1, NUM_CLASSES):
        s = agg['img_det'][c]
        tp, fp, fn = s['TP'], s['FP'], s['FN']
        total_tp += tp; total_fp += fp; total_fn += fn
        r = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        p = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else float('nan')
        img_det_results[c] = {
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': s['TN'],
            'recall': None if r != r else float(r),
            'precision': None if p != p else float(p),
            'f1': None if f1 != f1 else float(f1),
        }
        if r == r:
            valid_r.append(r)
        if p == p:
            valid_p.append(p)
    macro_recall = float(np.mean(valid_r)) if valid_r else 0.0
    macro_precision = float(np.mean(valid_p)) if valid_p else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else None
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else None

    return {
        'num_samples': agg['n_samples'],
        'total_time': agg['total_time'],
        'avg_inference_time': agg['total_time'] / max(agg['n_samples'], 1),
        'mIoU': float(miou),
        'mean_recall': macro_recall,         # macro (team naming)
        'mean_precision': macro_precision,   # macro
        'micro_recall': micro_recall,
        'micro_precision': micro_precision,
        'per_class_iou': {str(k): None if v != v else float(v)
                           for k, v in per_class_iou_global.items()},
        'image_level_detection': {str(k): v for k, v in img_det_results.items()},
        # For shard merging: raw intersection/union per class
        '_pixel_inter_union': {str(c): {
            'inter': int(agg['total_inter'][c]),
            'union': int(agg['total_union'][c]),
        } for c in range(NUM_CLASSES)},
    }


# ---------------------------------------------------------------------------
# Merge step (combines per-shard outputs into final files)
# ---------------------------------------------------------------------------

def merge_shards(output_dir, num_shards):
    print(f'[merge] combining {num_shards} shards in {output_dir}')
    for view_count, level_name in SNAPSHOT_LEVELS:
        # Merge records JSONLs
        final_records = output_dir / f'{level_name}_records.jsonl'
        shard_paths = [output_dir / f'{level_name}_records.shard{i}of{num_shards}.jsonl'
                       for i in range(num_shards)]
        missing = [str(p) for p in shard_paths if not p.exists()]
        if missing:
            print(f'[merge] WARNING — missing shard files: {missing}')
            continue
        with open(final_records, 'w') as out_f:
            for sp in shard_paths:
                with open(sp) as in_f:
                    for line in in_f:
                        out_f.write(line)
        print(f'[merge] wrote {final_records}')

        # Merge summaries by aggregating raw counts
        agg = {
            'total_inter': defaultdict(int),
            'total_union': defaultdict(int),
            'img_det': {c: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
                        for c in range(1, NUM_CLASSES)},
            'n_samples': 0,
            'total_time': 0.0,
        }
        for i in range(num_shards):
            sp = output_dir / f'{level_name}_summary.shard{i}of{num_shards}.json'
            if not sp.exists():
                print(f'[merge] WARNING — missing {sp}')
                continue
            with open(sp) as f:
                s = json.load(f)
            for c_str, iu in s['_pixel_inter_union'].items():
                c = int(c_str)
                agg['total_inter'][c] += iu['inter']
                agg['total_union'][c] += iu['union']
            for c_str, d in s['image_level_detection'].items():
                c = int(c_str)
                for k in ('TP', 'FP', 'FN', 'TN'):
                    agg['img_det'][c][k] += d[k]
            agg['n_samples'] += s['num_samples']
            agg['total_time'] += s['total_time']

        summary = build_summary_dict(agg)
        # Drop the internal field from the final merged summary
        summary.pop('_pixel_inter_union', None)
        final_sum = output_dir / f'{level_name}_summary.json'
        with open(final_sum, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'[merge] wrote {final_sum}')
        print(f'        n_samples={summary["num_samples"]}  '
              f'mIoU={summary["mIoU"]*100:.2f}%  '
              f'micro_recall={summary["micro_recall"]*100:.2f}%  '
              f'micro_precision={summary["micro_precision"]*100:.2f}%')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--test-json')
    parser.add_argument('--blade-filter-mask-dir', default=None)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int, default=None,
                        help='(debug) only process first N test samples')
    parser.add_argument('--save-pred-masks', action='store_true',
                        help='Also save pred mask PNGs per snapshot (~30-50 GB)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers for parallel image loading')
    parser.add_argument('--shard-id', type=int, default=0,
                        help='Which shard this process handles')
    parser.add_argument('--num-shards', type=int, default=1,
                        help='Total number of shards (split test set evenly)')
    parser.add_argument('--merge', action='store_true',
                        help='Run merge step only (combine per-shard outputs)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- merge-only mode ---------------------------------------------------
    if args.merge:
        if args.num_shards < 2:
            raise SystemExit('--merge requires --num-shards >= 2')
        merge_shards(output_dir, args.num_shards)
        return

    # ---- inference mode ---------------------------------------------------
    for arg in ('config', 'checkpoint', 'test_json'):
        if getattr(args, arg) is None:
            raise SystemExit(f'--{arg.replace("_", "-")} is required for inference')

    print(f'[setup] shard {args.shard_id}/{args.num_shards}  '
          f'device={args.device}  num_workers={args.num_workers}')
    print(f'[setup] loading config + model')
    cfg = Config.fromfile(args.config)
    init_default_scope('mmseg')
    model = MODELS.build(cfg.model)
    model = model.to(args.device).eval()

    print(f'[setup] loading checkpoint {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
    print(f'  missing keys: {len(missing)} | unexpected: {len(unexpected)} '
          f'| resumed iter: {ckpt.get("meta", {}).get("iter")}')
    del ckpt
    torch.cuda.empty_cache()
    data_preprocessor = model.data_preprocessor

    with open(args.test_json) as f:
        test_data = json.load(f)
    if args.limit:
        test_data = test_data[:args.limit]

    # Shard split (interleaved → balanced wtg/scene mix per shard)
    if args.num_shards > 1:
        test_data = [d for i, d in enumerate(test_data)
                     if i % args.num_shards == args.shard_id]
    print(f'[setup] {len(test_data)} test samples for this shard')
    print(f'[setup] TTA snapshot view counts: '
          f'{[vc for vc, _ in SNAPSHOT_LEVELS]}')

    shard_suffix = (f'.shard{args.shard_id}of{args.num_shards}'
                    if args.num_shards > 1 else '')

    # Per-snapshot output state
    records_jf = {}
    pred_dirs = {}
    agg = {}
    processed_paths = set()

    for view_count, level_name in SNAPSHOT_LEVELS:
        records_path = output_dir / f'{level_name}_records{shard_suffix}.jsonl'
        if records_path.exists():
            # Resume: skip already-processed image paths
            with open(records_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        processed_paths.add(rec['image_path'])
                    except Exception:
                        pass
            n_existing = sum(1 for _ in open(records_path))
            print(f'[setup] resume: {records_path.name} has {n_existing} rows')
        records_jf[level_name] = open(records_path, 'a')
        pred_dirs[level_name] = output_dir / f'pred_masks_{level_name}{shard_suffix}'
        if args.save_pred_masks:
            pred_dirs[level_name].mkdir(parents=True, exist_ok=True)
        agg[level_name] = {
            'total_inter': defaultdict(int),
            'total_union': defaultdict(int),
            'img_det': {c: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
                        for c in range(1, NUM_CLASSES)},
            'n_samples': 0,
            'total_time': 0.0,
        }

    # If resumed, replay aggregates from existing JSONLs.
    if processed_paths:
        for view_count, level_name in SNAPSHOT_LEVELS:
            records_path = output_dir / f'{level_name}_records{shard_suffix}.jsonl'
            if not records_path.exists():
                continue
            with open(records_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    agg[level_name]['n_samples'] += 1
                    agg[level_name]['total_time'] += rec.get('inference_time', 0.0)
                    for c, d in rec['instance_detection'].items():
                        ci = int(c)
                        agg[level_name]['img_det'][ci]['TP'] += d['TP']
                        agg[level_name]['img_det'][ci]['FP'] += d['FP']
                        agg[level_name]['img_det'][ci]['FN'] += d['FN']
                        if not d['gt_has'] and not d['pred_has']:
                            agg[level_name]['img_det'][ci]['TN'] += 1
                    # Per-image inter/union can't be reconstructed from the
                    # record (we only stored per-image IoU). For a perfectly
                    # accurate post-resume mIoU, run from a clean dir.

    output_state = {
        'records_jf': records_jf,
        'pred_dirs': pred_dirs,
        'agg': agg,
    }

    # DataLoader: parallel image loading
    samples_to_run = [d for d in test_data
                      if d.get('image_path') not in processed_paths]
    print(f'[setup] {len(samples_to_run)} samples to score '
          f'({len(test_data) - len(samples_to_run)} skipped via resume)')

    dataset = TestImageDataset(samples_to_run, args.blade_filter_mask_dir)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False,
        collate_fn=_passthrough_collate,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=False,
    )

    t0 = time.time()
    skipped_missing = 0
    pbar = tqdm(total=len(samples_to_run),
                desc=f'TTA shard{args.shard_id}', ncols=100)
    for prepared in loader:
        if prepared is None:
            skipped_missing += 1
            pbar.update(1)
            continue
        try:
            process_image(model, data_preprocessor, prepared, args, output_state)
        except Exception as e:
            print(f'\n[warn] failed on {prepared.get("image_path")}: {e}')
            skipped_missing += 1
        pbar.update(1)
    pbar.close()

    for jf in records_jf.values():
        jf.close()

    elapsed = time.time() - t0
    print(f'\n[done] shard{args.shard_id}: elapsed {elapsed/60:.1f} min  '
          f'missing-skip: {skipped_missing}')

    # Write per-shard summaries (used later by --merge)
    print('\n' + '=' * 60)
    for view_count, level_name in SNAPSHOT_LEVELS:
        summary_path = output_dir / f'{level_name}_summary{shard_suffix}.json'
        s = build_summary_dict(agg[level_name])
        with open(summary_path, 'w') as f:
            json.dump(s, f, indent=2)
        labels = [TTA_VIEWS[i]['label'] for i in range(view_count)]
        print(f'\n=== shard{args.shard_id} {level_name.upper()} '
              f'(views={view_count}: {", ".join(labels)}) ===')
        print(f'  n_samples         : {s["num_samples"]}')
        print(f'  mIoU              : {s["mIoU"]*100:.2f}%')
        if s.get("mean_recall") is not None:
            print(f'  Macro recall      : {s["mean_recall"]*100:.2f}%')
        if s.get("mean_precision") is not None:
            print(f'  Macro precision   : {s["mean_precision"]*100:.2f}%')
        if s.get("micro_recall") is not None:
            print(f'  Micro recall      : {s["micro_recall"]*100:.2f}%')
        if s.get("micro_precision") is not None:
            print(f'  Micro precision   : {s["micro_precision"]*100:.2f}%')
        print(f'  Saved             : {summary_path}')

    if args.num_shards > 1:
        print(f'\n[info] this was shard {args.shard_id} of {args.num_shards}; '
              f'run with --merge once all shards finish to combine.')


if __name__ == '__main__':
    main()
