# Copyright (c) OpenMMLab. All rights reserved.
"""
Faster validation hook for DMDD SAM 3.1 LoRA run.

Same metrics, same best-checkpoint logic, same visualization layout as
``FastValHook`` (mmseg/engine/hooks/fast_val_hook.py).

Differences:
  1. Image / GT / blade-mask loading runs in parallel DataLoader workers,
     so the GPU stops waiting on cv2.imread of 8K images between forwards.
  2. Inference is wrapped in ``torch.autocast(cuda, bfloat16)``, halving
     per-window cost on H200/A100 without affecting ranking.
  3. ``_blade_aware_slide_inference`` batches non-empty crop windows
     (``slide_batch_size`` per forward), turning ~100 single-window kernel
     launches into ~12 batched ones. Mathematically equivalent modulo float
     non-associativity (~1e-5, well below ckpt-selection sensitivity).

The full val set is still used — no subsampling, no stride change. Best
checkpoint selection (``best_recall_iter_*.pth`` / ``best_mIoU_iter_*.pth``)
is preserved relative to the original hook (modulo bf16 rounding).

Usage in config:
    custom_imports = dict(
        imports=['sam3_backbone', 'fast_val_hook_v2'],
        allow_failed_imports=False,
    )
    custom_hooks = [
        dict(type='FastValHookV2',
             data_json='/path/to/val.json',
             val_interval=10000,
             num_workers=4,         # NEW
             prefetch_factor=2,     # NEW
             use_bf16=True,         # NEW
             slide_batch_size=8,    # NEW
             blade_filter_mask_dir='/path/to/blade_masks_640'),
    ]
    val_cfg = None
    val_dataloader = None
    val_evaluator = None
"""

import contextlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from tqdm import tqdm

from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmseg.registry import HOOKS

sys.path.insert(0, '/home/work/workspace/jongwon/dmdd/scripts')


# ---------------------------------------------------------------------------
# Module-level helpers (must be picklable for DataLoader workers).
# ---------------------------------------------------------------------------

def _load_blade_mask_full_res(img_path, target_hw, mask_dir, classes):
    """Load and upsample (nearest) the blade mask for one image.

    Layout: ``<mask_dir>/<wtg>/<stem>.png`` at native ~640 long edge.
    Returns binary uint8 (h, w) where 1 = blade-or-nose. ``None`` if missing.
    Standalone (not a method) so DataLoader workers can pickle it.
    """
    wtg = Path(img_path).parent.name
    stem = Path(img_path).stem
    mask_path = Path(mask_dir) / wtg / f'{stem}.png'
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    blade = np.isin(mask, classes).astype(np.uint8)
    h, w = target_hw
    if blade.shape != (h, w):
        blade = cv2.resize(blade, (w, h), interpolation=cv2.INTER_NEAREST)
    return blade


class _FastValDataset(torch.utils.data.Dataset):
    """Reads image + GT mask + (optional) blade mask. One sample per __getitem__.

    Returns ``None`` for missing files; the collate_fn passes that through so
    the main loop can skip it.
    """

    def __init__(self, samples, blade_filter_mask_dir, blade_filter_classes):
        self.samples = samples
        self.blade_filter_mask_dir = blade_filter_mask_dir
        self.blade_filter_classes = tuple(blade_filter_classes)

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
        if self.blade_filter_mask_dir is not None:
            blade_mask = _load_blade_mask_full_res(
                image_path, image.shape[:2],
                self.blade_filter_mask_dir, self.blade_filter_classes,
            )

        return {
            'image': image,
            'gt_mask': gt_mask,
            'blade_mask': blade_mask,
            'image_path': image_path,
            'idx': idx,
        }


def _passthrough_collate(batch):
    # batch_size=1; we return the single dict (or None) unchanged.
    return batch[0]


# ---------------------------------------------------------------------------
# Hook
# ---------------------------------------------------------------------------

@HOOKS.register_module()
class FastValHookV2(Hook):
    """Parallelized + bf16 variant of FastValHook.

    Args:
        data_json: Path to validation data JSON file.
        val_interval: Run validation every N iterations.
        num_classes: Number of classes (default 9: BG + 8 defects).
        blade_mask_dir: Optional legacy blade mask directory (ApplyBladeMask-
            style, ignore_index post-processing).
        blade_mask_split: Split name for the legacy blade mask loader.
        blade_filter_mask_dir: Optional new-style blade filter mask dir, used
            for slide-window skipping and post-mask. Mirror layout under
            ``<dir>/<wtg>/<stem>.png``.
        blade_filter_classes: Class IDs in the blade mask file that count as
            "blade" (default (1, 2) = blade + nose).
        save_vis_samples: Number of visualization samples to save per val run
            (default 10; the original hardcodes 100 internally — keep parity).
        initial_val: If True, run validation once before training starts.
        num_workers: DataLoader workers for image / mask I/O (default 4).
        prefetch_factor: DataLoader prefetch_factor (default 2). Ignored when
            num_workers == 0.
        use_bf16: Wrap inference in ``torch.autocast(cuda, bfloat16)``
            (default True).
        persistent_workers: Keep DataLoader workers alive between val runs
            (default True). Saves the ~5-10s worker spin-up each val.
        slide_batch_size: How many crop windows to pack into a single
            ``model.encode_decode`` forward. Default 8 (~2-4GB extra VRAM on
            H200). Set to 1 to recover the per-window behavior of the
            original FastValHook for A/B testing.
        cc_downsample_factor: Integer downsample factor applied to pred + GT
            masks before ``cv2.connectedComponentsWithStats``. Default 1
            (no downsample, identical to original). Set to 2 or 4 to speed
            up the image-level-detection pass at the cost of possibly
            missing very small defects. Note: only affects image-level
            TP/FP/FN counting; pixel-IoU is always computed on full-res
            masks.
    """

    priority = 'LOW'

    CLASS_NAMES = [
        "Background", "La Exposure", "La Damage", "La Crack",
        "La Open", "Bond Crack", "Bond Open", "Receptor Lightning",
        "Receptor Damage"
    ]

    CLASS_COLORS_RGB = [
        (0, 0, 0),         # 0: Background
        (255, 255, 0),     # 1: La Exposure - Yellow
        (0, 255, 0),       # 2: La Damage - Green
        (0, 200, 0),       # 3: La Crack - Dark Green
        (0, 150, 100),     # 4: La Open - Teal
        (0, 255, 255),     # 5: Bond Crack - Cyan
        (0, 200, 255),     # 6: Bond Open - Light Blue
        (128, 0, 255),     # 7: Receptor Lightning - Purple
        (255, 0, 128),     # 8: Receptor Damage - Pink
    ]

    # Cache the BGR uint8 representation once per class (was per-image
    # `np.array([color[2], color[1], color[0]])` allocation in the viz
    # overlay loop). Index by cls_id.
    CLASS_COLORS_BGR_ARR = [
        np.array([c[2], c[1], c[0]], dtype=np.uint8)
        for c in CLASS_COLORS_RGB
    ]

    def __init__(
        self,
        data_json: str,
        val_interval: int = 10000,
        num_classes: int = 9,
        blade_mask_dir: Optional[str] = None,
        blade_mask_split: str = 'test',
        blade_filter_mask_dir: Optional[str] = None,
        blade_filter_classes: tuple = (1, 2),
        save_vis_samples: int = 10,
        initial_val: bool = True,
        num_workers: int = 8,
        prefetch_factor: int = 2,
        use_bf16: bool = True,
        persistent_workers: bool = True,
        slide_batch_size: int = 8,
        cc_downsample_factor: int = 1,
    ):
        self.data_json = data_json
        self.val_interval = val_interval
        self.initial_val = initial_val
        self.num_classes = num_classes
        self.blade_mask_dir = blade_mask_dir
        self.blade_mask_split = blade_mask_split
        self.blade_filter_mask_dir = blade_filter_mask_dir
        self.blade_filter_classes = tuple(blade_filter_classes)
        self.save_vis_samples = save_vis_samples

        self.num_workers = int(num_workers)
        self.prefetch_factor = int(prefetch_factor)
        self.use_bf16 = bool(use_bf16)
        self.persistent_workers = bool(persistent_workers)
        self.slide_batch_size = max(1, int(slide_batch_size))
        self.cc_downsample_factor = max(1, int(cc_downsample_factor))

        with open(data_json, 'r') as f:
            self.data = json.load(f)
        print_log(
            f'FastValHookV2: Loaded {len(self.data)} samples from {data_json}',
            logger='current')

        self.blade_mask_loader = None
        if blade_mask_dir:
            from blade_mask_utils import BladeMaskLoader
            self.blade_mask_loader = BladeMaskLoader(
                blade_mask_dir=blade_mask_dir,
                split=blade_mask_split,
                ignore_index=255,
            )
            print_log(
                f'FastValHookV2: Blade mask enabled '
                f'({blade_mask_dir}/{blade_mask_split})',
                logger='current')

        print_log(
            f'FastValHookV2: num_workers={self.num_workers}, '
            f'prefetch_factor={self.prefetch_factor}, '
            f'use_bf16={self.use_bf16}, '
            f'persistent_workers={self.persistent_workers}, '
            f'slide_batch_size={self.slide_batch_size}',
            logger='current')

        self.best_miou = 0.0
        self.best_mean_recall = 0.0
        self.val_history = []
        self._loader = None  # lazily built; reused across val runs

    # -- DataLoader lifecycle ------------------------------------------------

    def _get_loader(self):
        if self._loader is not None:
            return self._loader
        dataset = _FastValDataset(
            self.data, self.blade_filter_mask_dir, self.blade_filter_classes,
        )
        kwargs = dict(
            dataset=dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            collate_fn=_passthrough_collate,
        )
        if self.num_workers > 0:
            kwargs['prefetch_factor'] = self.prefetch_factor
            kwargs['persistent_workers'] = self.persistent_workers
        self._loader = torch.utils.data.DataLoader(**kwargs)
        return self._loader

    # -- Hook entry points ---------------------------------------------------

    def before_train(self, runner):
        if self.initial_val:
            self._run_validation(runner)
        else:
            print_log(
                'FastValHookV2: Skipping initial validation (initial_val=False)',
                logger='current')

    def after_train_iter(self, runner, batch_idx: int, data_batch=None,
                          outputs=None):
        if (runner.iter + 1) % self.val_interval != 0:
            return
        self._run_validation(runner)

    # -- Core validation -----------------------------------------------------

    def _run_validation(self, runner):
        from mmseg.structures import SegDataSample

        # DDP NOTE: we deliberately do NOT rank-guard here. A naive
        # "rank-0 runs val, rank-N barriers" pattern hits the NCCL barrier
        # timeout (~30 min default) when val takes hours. Letting both ranks
        # run val redundantly burns GPU 1's compute during val but avoids
        # the sync issue. ``runner.save_checkpoint`` is already
        # ``@master_only`` in mmengine, so only rank 0 actually writes
        # ``best_recall_*.pth`` / ``best_mIoU_*.pth`` to disk.

        current_iter = runner.iter + 1
        work_dir = Path(runner.work_dir)
        val_dir = work_dir / 'val_results' / f'iter_{current_iter}'
        val_dir.mkdir(parents=True, exist_ok=True)
        vis_dir = val_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)

        print_log(f'\n{"="*60}', logger='current')
        print_log(
            f'FastValHookV2: Running validation at iter {current_iter}',
            logger='current')
        print_log(f'{"="*60}', logger='current')

        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        model.eval()
        data_preprocessor = model.data_preprocessor

        total_intersection = defaultdict(int)
        total_union = defaultdict(int)
        total_time = 0.0
        skipped = 0

        img_det = {cls_id: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
                   for cls_id in range(1, self.num_classes)}

        # Honor self.save_vis_samples (defaults 10). Originally FastValHook
        # hardcoded 100 here regardless of the constructor argument.
        max_vis_samples = max(0, int(self.save_vis_samples))

        loader = self._get_loader()
        amp_ctx = (torch.autocast('cuda', dtype=torch.bfloat16)
                   if self.use_bf16 else contextlib.nullcontext())

        n_total = len(self.data)
        for sample in tqdm(loader, total=n_total, desc='FastValV2', ncols=80):
            if sample is None:
                skipped += 1
                continue

            image = sample['image']
            gt_mask = sample['gt_mask']
            blade_mask_full = sample['blade_mask']
            image_path = sample['image_path']
            idx = sample['idx']

            start = time.time()
            with torch.no_grad():
                img_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
                img_tensor = img_tensor.unsqueeze(0).cuda(non_blocking=True)

                data_sample = SegDataSample()
                data_sample.set_metainfo({
                    'img_shape': image.shape[:2],
                    'ori_shape': image.shape[:2],
                    'pad_shape': image.shape[:2],
                    'scale_factor': (1.0, 1.0),
                })

                data = {'inputs': img_tensor, 'data_samples': [data_sample]}
                data = data_preprocessor(data, False)

                with amp_ctx:
                    if blade_mask_full is not None:
                        seg_logits = self._blade_aware_slide_inference(
                            model, data['inputs'], data['data_samples'],
                            blade_mask_full)
                        pred_mask = seg_logits.argmax(dim=1)[0].cpu().numpy(
                        ).astype(np.uint8)
                    else:
                        result = model.predict(
                            data['inputs'], data['data_samples'])[0]
                        pred_mask = result.pred_sem_seg.data.cpu().numpy()[0]

            total_time += time.time() - start

            if blade_mask_full is not None:
                pred_mask = pred_mask.copy()
                pred_mask[blade_mask_full == 0] = 0

            if self.blade_mask_loader:
                pred_mask, gt_mask = self.blade_mask_loader.apply(
                    pred_mask, gt_mask, image_path)
                iou_result = self._calculate_iou_with_ignore(
                    pred_mask, gt_mask, ignore_index=255)
            else:
                iou_result = self._calculate_iou(pred_mask, gt_mask)

            for cls_id in range(self.num_classes):
                total_intersection[cls_id] += iou_result['intersection'][cls_id]
                total_union[cls_id] += iou_result['union'][cls_id]

            # Image-level detection: optional pre-downsample for CC.
            # Default cc_downsample_factor=1 → identical to original.
            ds = self.cc_downsample_factor
            if ds > 1:
                h0, w0 = pred_mask.shape[:2]
                pred_for_cc = cv2.resize(
                    pred_mask, (w0 // ds, h0 // ds),
                    interpolation=cv2.INTER_NEAREST)
                gt_for_cc = cv2.resize(
                    gt_mask, (w0 // ds, h0 // ds),
                    interpolation=cv2.INTER_NEAREST)
            else:
                pred_for_cc = pred_mask
                gt_for_cc = gt_mask

            # ONE full-res scan each to find which classes are actually
            # present; cheaper than 8 per-class `==cls_id` allocations.
            gt_classes = set(np.unique(gt_for_cc).tolist())
            pred_classes = set(np.unique(pred_for_cc).tolist())

            for cls_id in range(1, self.num_classes):
                gt_has = cls_id in gt_classes
                pred_has = cls_id in pred_classes
                if not (gt_has or pred_has):
                    # Neither present → TN, skip the full-res scans entirely.
                    img_det[cls_id]['TN'] += 1
                    continue
                overlap = False
                if gt_has and pred_has:
                    gt_cls = (gt_for_cc == cls_id).astype(np.uint8)
                    pred_cls = (pred_for_cc == cls_id).astype(np.uint8)
                    _, _, gt_stats, _ = cv2.connectedComponentsWithStats(
                        gt_cls, connectivity=8)
                    _, _, pred_stats, _ = cv2.connectedComponentsWithStats(
                        pred_cls, connectivity=8)
                    for i in range(1, len(pred_stats)):
                        px, py, pw, ph = pred_stats[i, :4]
                        for j in range(1, len(gt_stats)):
                            gx, gy, gw, gh = gt_stats[j, :4]
                            if (px < gx + gw and gx < px + pw
                                    and py < gy + gh and gy < py + ph):
                                overlap = True
                                break
                        if overlap:
                            break
                if overlap:
                    img_det[cls_id]['TP'] += 1
                else:
                    if pred_has:
                        img_det[cls_id]['FP'] += 1
                    if gt_has:
                        img_det[cls_id]['FN'] += 1
                    if not gt_has and not pred_has:
                        img_det[cls_id]['TN'] += 1

            if idx < max_vis_samples:
                sample_name = Path(image_path).stem
                vis = self._create_full_visualization(
                    image, pred_mask, gt_mask)
                if vis is not None:
                    vis_path = vis_dir / f'{idx:04d}_{sample_name}.jpg'
                    cv2.imwrite(str(vis_path), vis)

            del image, gt_mask, pred_mask

        if skipped:
            print_log(
                f'FastValHookV2: Skipped {skipped} samples with missing files',
                logger='current')

        n_eval = n_total - skipped

        # ---- per-class IoU ----
        per_class_iou = {}
        valid_ious = []
        for cls_id in range(self.num_classes):
            if total_union[cls_id] > 0:
                iou = total_intersection[cls_id] / total_union[cls_id]
                per_class_iou[cls_id] = iou
                valid_ious.append(iou)
            else:
                per_class_iou[cls_id] = float('nan')
        miou = np.nanmean(valid_ious) if valid_ious else 0.0

        # ---- image-level recall / precision ----
        img_det_results = {}
        valid_recalls = []
        valid_precisions = []
        for cls_id in range(1, self.num_classes):
            s = img_det[cls_id]
            tp, fp, fn = s['TP'], s['FP'], s['FN']
            recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
            precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else float('nan'))
            img_det_results[cls_id] = {
                'TP': tp, 'FP': fp, 'FN': fn, 'TN': s['TN'],
                'recall': recall, 'precision': precision, 'f1': f1,
            }
            if not np.isnan(recall):
                valid_recalls.append(recall)
            if not np.isnan(precision):
                valid_precisions.append(precision)
        mean_recall = np.mean(valid_recalls) if valid_recalls else 0.0
        mean_precision = np.mean(valid_precisions) if valid_precisions else 0.0

        # ---- log ----
        print_log(f'\nFastValV2 Results (iter {current_iter}):',
                  logger='current')
        avg = total_time / max(n_eval, 1)
        print_log(
            f'  Samples: {n_eval} (skipped {skipped}), '
            f'Time: {total_time:.1f}s ({avg:.2f}s/img)',
            logger='current')
        print_log(f'\n  {"Class":<20} {"IoU":>10}', logger='current')
        print_log(f'  {"-"*32}', logger='current')
        for cls_id in range(self.num_classes):
            cls_name = (self.CLASS_NAMES[cls_id]
                        if cls_id < len(self.CLASS_NAMES)
                        else f'Class_{cls_id}')
            iou = per_class_iou[cls_id]
            if np.isnan(iou):
                print_log(f'  {cls_name:<20} {"N/A":>10}', logger='current')
            else:
                print_log(f'  {cls_name:<20} {iou*100:>9.2f}%',
                          logger='current')
        print_log(f'  {"-"*32}', logger='current')
        print_log(f'  {"mIoU":<20} {miou*100:>9.2f}%', logger='current')

        print_log(f'\n  Image-Level Detection:', logger='current')
        print_log(
            f'  {"Class":<20} {"TP":>5} {"FP":>5} {"FN":>5} '
            f'{"Recall":>10} {"Prec":>10}',
            logger='current')
        print_log(f'  {"-"*58}', logger='current')
        for cls_id in range(1, self.num_classes):
            cls_name = (self.CLASS_NAMES[cls_id]
                        if cls_id < len(self.CLASS_NAMES)
                        else f'Class_{cls_id}')
            r = img_det_results[cls_id]
            recall_str = (f'{r["recall"]*100:.1f}%'
                          if not np.isnan(r['recall']) else 'N/A')
            prec_str = (f'{r["precision"]*100:.1f}%'
                        if not np.isnan(r['precision']) else 'N/A')
            print_log(
                f'  {cls_name:<20} {r["TP"]:>5} {r["FP"]:>5} {r["FN"]:>5} '
                f'{recall_str:>10} {prec_str:>10}',
                logger='current')
        print_log(f'  {"-"*58}', logger='current')
        print_log(
            f'  {"Mean":<20} {"":>5} {"":>5} {"":>5} '
            f'{mean_recall*100:>9.1f}% {mean_precision*100:>9.1f}%',
            logger='current')

        # ---- best-checkpoint tracking (identical to FastValHook) ----
        is_best_recall = bool(mean_recall > self.best_mean_recall)
        if is_best_recall:
            self.best_mean_recall = mean_recall
            print_log(
                f'\n  ** New best Mean Recall: {mean_recall*100:.2f}% **',
                logger='current')
            best_ckpt_path = work_dir / f'best_recall_iter_{current_iter}.pth'
            runner.save_checkpoint(
                str(work_dir),
                filename=f'best_recall_iter_{current_iter}.pth',
                save_optimizer=False)
            for old_ckpt in work_dir.glob('best_recall_iter_*.pth'):
                if old_ckpt.name != f'best_recall_iter_{current_iter}.pth':
                    old_ckpt.unlink()
                    print_log(f'  Removed old best: {old_ckpt.name}',
                              logger='current')
            print_log(f'  Saved best checkpoint: {best_ckpt_path.name}',
                      logger='current')

        is_best_miou = bool(miou > self.best_miou)
        if is_best_miou:
            self.best_miou = miou
            print_log(f'\n  ** New best mIoU: {miou*100:.2f}% **',
                      logger='current')
            best_miou_ckpt_path = (
                work_dir / f'best_mIoU_iter_{current_iter}.pth')
            runner.save_checkpoint(
                str(work_dir),
                filename=f'best_mIoU_iter_{current_iter}.pth',
                save_optimizer=False)
            for old_ckpt in work_dir.glob('best_mIoU_iter_*.pth'):
                if old_ckpt.name != f'best_mIoU_iter_{current_iter}.pth':
                    old_ckpt.unlink()
                    print_log(f'  Removed old best: {old_ckpt.name}',
                              logger='current')
            print_log(f'  Saved best checkpoint: {best_miou_ckpt_path.name}',
                      logger='current')

        print_log(
            f'\n  Best Mean Recall so far: {self.best_mean_recall*100:.2f}%',
            logger='current')
        print_log(f'  Best mIoU so far: {self.best_miou*100:.2f}%',
                  logger='current')

        # ---- json dump ----
        results = {
            'iter': current_iter,
            'num_samples': n_eval,
            'skipped': skipped,
            'total_time': total_time,
            'avg_time_per_sample': avg,
            'mIoU': float(miou),
            'mean_recall': float(mean_recall),
            'mean_precision': float(mean_precision),
            'is_best_recall': is_best_recall,
            'is_best_miou': is_best_miou,
            'per_class_iou': {
                self.CLASS_NAMES[k]: (float(v) if not np.isnan(v) else None)
                for k, v in per_class_iou.items()
            },
            'image_level_detection': {
                self.CLASS_NAMES[cls_id]: {
                    'TP': r['TP'], 'FP': r['FP'], 'FN': r['FN'],
                    'recall': (float(r['recall'])
                               if not np.isnan(r['recall']) else None),
                    'precision': (float(r['precision'])
                                  if not np.isnan(r['precision']) else None),
                    'f1': (float(r['f1'])
                           if not np.isnan(r['f1']) else None),
                }
                for cls_id, r in img_det_results.items()
            },
        }
        with open(val_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

        self.val_history.append({
            'iter': current_iter,
            'mIoU': float(miou),
            'mean_recall': float(mean_recall),
            'mean_precision': float(mean_precision),
            'per_class_iou': {
                k: (float(v) if not np.isnan(v) else None)
                for k, v in per_class_iou.items()
            },
        })
        with open(work_dir / 'val_history.json', 'w') as f:
            json.dump(self.val_history, f, indent=2)

        print_log(
            f'\n  Visualizations saved: {min(max_vis_samples, n_eval)} samples',
            logger='current')
        print_log(f'  Results saved to: {val_dir}', logger='current')
        print_log(f'{"="*60}\n', logger='current')

        runner.message_hub.update_scalar('val/mIoU', miou)
        runner.message_hub.update_scalar('val/mean_recall', mean_recall)
        runner.message_hub.update_scalar('val/mean_precision', mean_precision)
        for cls_id, iou in per_class_iou.items():
            if not np.isnan(iou):
                cls_name = (self.CLASS_NAMES[cls_id]
                            if cls_id < len(self.CLASS_NAMES)
                            else f'Class_{cls_id}')
                runner.message_hub.update_scalar(f'val/{cls_name}_IoU', iou)
        for cls_id, r in img_det_results.items():
            cls_name = (self.CLASS_NAMES[cls_id]
                        if cls_id < len(self.CLASS_NAMES)
                        else f'Class_{cls_id}')
            if not np.isnan(r['recall']):
                runner.message_hub.update_scalar(
                    f'val/{cls_name}_recall', r['recall'])
            if not np.isnan(r['precision']):
                runner.message_hub.update_scalar(
                    f'val/{cls_name}_precision', r['precision'])

        model.train()

    # -- helpers (copied verbatim from FastValHook) --------------------------

    def _create_full_visualization(self, image: np.ndarray,
                                    pred_mask: np.ndarray,
                                    gt_mask: np.ndarray, alpha: float = 0.5,
                                    max_width: int = 2000
                                    ) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        # ONE np.unique scan per mask instead of 8 per-class `== cls_id` scans;
        # CLASS_COLORS_BGR_ARR is precomputed at class level (no per-image
        # allocation of the BGR tuple).
        pred_classes_present = set(np.unique(pred_mask).tolist()) - {0}
        gt_classes_present = set(np.unique(gt_mask).tolist()) - {0}

        pred_overlay = image.copy()
        for cls_id in pred_classes_present:
            if not (1 <= cls_id < self.num_classes):
                continue
            pred_cls_mask = (pred_mask == cls_id)
            color_bgr = self.CLASS_COLORS_BGR_ARR[cls_id]
            pred_overlay[pred_cls_mask] = (
                (1 - alpha) * image[pred_cls_mask] + alpha * color_bgr
            ).astype(np.uint8)

        gt_overlay = image.copy()
        for cls_id in gt_classes_present:
            if not (1 <= cls_id < self.num_classes):
                continue
            gt_cls_mask = (gt_mask == cls_id)
            color_bgr = self.CLASS_COLORS_BGR_ARR[cls_id]
            gt_overlay[gt_cls_mask] = (
                (1 - alpha) * image[gt_cls_mask] + alpha * color_bgr
            ).astype(np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(1.0, min(h, w) / 1000)
        thickness = max(2, int(font_scale * 2))
        cv2.putText(pred_overlay, 'Prediction', (20, 60),
                    font, font_scale, (255, 255, 255), thickness + 2)
        cv2.putText(pred_overlay, 'Prediction', (20, 60),
                    font, font_scale, (0, 255, 0), thickness)
        cv2.putText(gt_overlay, 'Ground Truth', (20, 60),
                    font, font_scale, (255, 255, 255), thickness + 2)
        cv2.putText(gt_overlay, 'Ground Truth', (20, 60),
                    font, font_scale, (0, 255, 0), thickness)
        vis = np.concatenate([pred_overlay, gt_overlay], axis=1)
        if vis.shape[1] > max_width:
            scale = max_width / vis.shape[1]
            vis = cv2.resize(vis, None, fx=scale, fy=scale)
        return vis

    def _blade_aware_slide_inference(self, model, inputs, data_samples,
                                       blade_mask):
        """Batched slide inference, skipping windows with 0 blade pixels.

        Phase 1 collects all non-empty window coords via a CPU-side check
        against the numpy ``blade_mask`` (no GPU syncs).
        Phase 2 packs windows into batches of ``self.slide_batch_size`` and
        runs each batch through ``model.encode_decode`` in one forward.

        Mathematically equivalent to the per-window loop modulo float
        non-associativity (~1e-5).

        Args:
            model: EncoderDecoder.
            inputs: (1, 3, H, W) preprocessed input tensor on GPU.
            data_samples: list[SegDataSample] (length 1).
            blade_mask: numpy uint8 (H, W) on CPU. 1 = blade-or-nose.

        Returns:
            seg_logits: (1, num_classes, H, W) tensor on GPU.
        """
        test_cfg = model.test_cfg
        h_stride, w_stride = test_cfg['stride']
        h_crop, w_crop = test_cfg['crop_size']
        batch_size, _, h_img, w_img = inputs.shape
        out_channels = model.decode_head.out_channels

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))

        # Phase 1: collect non-empty window coords (CPU-side, no GPU sync).
        coords = []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                if blade_mask[y1:y2, x1:x2].sum() == 0:
                    continue
                coords.append((y1, x1, y2, x2))

        if not coords:
            # Entire image non-blade; argmax over zeros gives background.
            return preds

        base_meta = dict(data_samples[0].metainfo)
        # IMPORTANT: mmseg's decode_head.predict_by_feat picks the
        # slide-inference resize branch *only when img_shape is a
        # torch.Size*. A Python tuple falls through to pad_shape
        # (which here = full original image, e.g. 8256×5504), causing
        # chunk_logits to come back at full image resolution and the
        # scatter-add to shape-mismatch. Use torch.Size to stay on the
        # crop-resolution path.
        base_meta['img_shape'] = torch.Size([h_crop, w_crop])

        # Phase 2: batched forward, scatter back into preds/count_mat.
        B = self.slide_batch_size
        for i in range(0, len(coords), B):
            chunk = coords[i:i + B]
            crops = torch.cat(
                [inputs[:, :, y1:y2, x1:x2] for (y1, x1, y2, x2) in chunk],
                dim=0,
            )  # (b, 3, h_crop, w_crop)
            metas = [dict(base_meta) for _ in chunk]
            chunk_logits = model.encode_decode(crops, metas)
            # chunk_logits: (b, num_classes, h_crop, w_crop)
            for j, (y1, x1, y2, x2) in enumerate(chunk):
                preds[:, :, y1:y2, x1:x2] += chunk_logits[j:j + 1]
                count_mat[:, :, y1:y2, x1:x2] += 1

        count_mat = count_mat.clamp(min=1)
        return preds / count_mat

    def _confusion_matrix(self, pred: np.ndarray, gt: np.ndarray,
                            ignore_index: Optional[int] = None) -> np.ndarray:
        """Build an NxN confusion matrix in a single pass via bincount.

        Replaces 9 boolean masks + 9 logical_and + 9 logical_or with one
        ``np.bincount(num_classes * gt + pred)``. Bit-identical results, just
        much less allocation. Roughly 5-10× faster on 8K masks.
        """
        n = int(self.num_classes)
        # Flatten and (optionally) drop ignored pixels.
        pred_flat = pred.astype(np.int64, copy=False).ravel()
        gt_flat = gt.astype(np.int64, copy=False).ravel()
        if ignore_index is not None:
            valid = gt_flat != ignore_index
            pred_flat = pred_flat[valid]
            gt_flat = gt_flat[valid]
        # Guard: clip out-of-range values to background (255 etc. would
        # otherwise produce out-of-bounds bincount keys).
        np.clip(pred_flat, 0, n - 1, out=pred_flat)
        np.clip(gt_flat, 0, n - 1, out=gt_flat)
        cm = np.bincount(n * gt_flat + pred_flat, minlength=n * n)[:n * n]
        return cm.reshape(n, n)

    def _calculate_iou(self, pred: np.ndarray, gt: np.ndarray) -> Dict:
        n = self.num_classes
        cm = self._confusion_matrix(pred, gt)
        intersection = np.diag(cm)
        pred_total = cm.sum(axis=0)  # cols: pred
        gt_total = cm.sum(axis=1)    # rows: gt
        union = pred_total + gt_total - intersection
        return {
            'intersection': {i: int(intersection[i]) for i in range(n)},
            'union': {i: int(union[i]) for i in range(n)},
        }

    def _calculate_iou_with_ignore(self, pred: np.ndarray, gt: np.ndarray,
                                    ignore_index: int = 255) -> Dict:
        n = self.num_classes
        cm = self._confusion_matrix(pred, gt, ignore_index=ignore_index)
        intersection = np.diag(cm)
        pred_total = cm.sum(axis=0)
        gt_total = cm.sum(axis=1)
        union = pred_total + gt_total - intersection
        return {
            'intersection': {i: int(intersection[i]) for i in range(n)},
            'union': {i: int(union[i]) for i in range(n)},
        }
