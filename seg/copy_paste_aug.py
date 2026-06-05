"""mmseg transform: CopyPasteRareInstances.

Pastes randomly sampled rare-class instances onto the current training image
at on-blade locations. Implements feathered alpha blending so paste edges
aren't jagged. Updates `img` (BGR uint8) and `gt_seg_map` (uint8 class indices)
in place.

Why: v2 baseline has 73 % of FNs as "completely missed" — the rare classes
just aren't being seen enough in training. Copy-paste directly puts more
instances of those classes in front of the model.

Index format: see extract_instance_pool.py output.
"""
import os, json, random
from pathlib import Path
from typing import List, Dict

import numpy as np
import cv2

from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class CopyPasteRareInstances(BaseTransform):
    def __init__(
        self,
        pool_dir: str = "/home/work/workspace/azka5/dmdd_pipeline/data/instance_pool",
        blade_mask_dir: str = "/home/work/workspace/jongwon/dmdd/data/blade_masks_640",
        target_classes: List[int] = (3, 4, 5, 6, 7),     # rare ones
        prob: float = 0.7,                                # fraction of images aug'd
        n_pastes_range=(1, 3),                            # how many per image
        max_paste_frac: float = 0.10,                     # max bbox area / image area
        feather: int = 5,
        seed: int = None,
    ):
        super().__init__()
        self.pool_dir = pool_dir
        self.blade_mask_dir = blade_mask_dir
        self.target_classes = tuple(target_classes)
        self.prob = prob
        self.n_pastes_range = tuple(n_pastes_range)
        self.max_paste_frac = max_paste_frac
        self.feather = feather
        self.rng = random.Random(seed)

        idx_path = os.path.join(pool_dir, "index.jsonl")
        self.by_class: Dict[int, List[dict]] = {c: [] for c in self.target_classes}
        if os.path.isfile(idx_path):
            for line in open(idx_path):
                e = json.loads(line)
                c = e.get("class_id")
                if c in self.by_class:
                    self.by_class[c].append(e)
        self._total = sum(len(v) for v in self.by_class.values())
        print(f"[CopyPasteRareInstances] loaded {self._total} instances "
              f"across classes {self.target_classes}", flush=True)

    def _load_instance(self, entry):
        cdir = os.path.join(self.pool_dir, f"c{entry['class_id']}")
        ip = os.path.join(cdir, f"{entry['uid']}_img.png")
        mp = os.path.join(cdir, f"{entry['uid']}_msk.png")
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        msk = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        return img, msk

    def _load_blade_mask(self, image_path, H, W):
        """Read blade mask if available, scaled to (H,W). Otherwise all-ones."""
        try:
            stem = Path(image_path).stem
            wtg = Path(image_path).parent.name
            mp = os.path.join(self.blade_mask_dir, wtg, f"{stem}.png")
            if not os.path.isfile(mp):
                return np.ones((H, W), dtype=np.uint8)
            bm = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
            if bm.shape != (H, W):
                bm = cv2.resize(bm, (W, H), interpolation=cv2.INTER_NEAREST)
            return (bm > 0).astype(np.uint8)
        except Exception:
            return np.ones((H, W), dtype=np.uint8)

    def _paste_one(self, img, gt, blade_mask, entry):
        ins_img, ins_msk = self._load_instance(entry)
        if ins_img is None or ins_msk is None:
            return
        ih, iw = ins_msk.shape
        H, W = gt.shape

        # Resize to keep at <= max_paste_frac of image area, otherwise downscale
        target_area = self.max_paste_frac * H * W
        cur_area = ih * iw
        scale = min(1.0, (target_area / max(cur_area, 1)) ** 0.5)
        # Also randomize scale a bit
        scale *= self.rng.uniform(0.6, 1.0)
        if scale < 0.99:
            new_w = max(8, int(iw * scale)); new_h = max(8, int(ih * scale))
            ins_img = cv2.resize(ins_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            ins_msk = cv2.resize(ins_msk, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            ih, iw = new_h, new_w

        if ih > H or iw > W:
            return

        # Find a paste location whose center is on-blade
        for _ in range(20):
            cx = self.rng.randint(iw // 2, W - iw // 2 - 1)
            cy = self.rng.randint(ih // 2, H - ih // 2 - 1)
            if blade_mask[cy, cx] > 0:
                x0 = cx - iw // 2; y0 = cy - ih // 2
                break
        else:
            return  # no on-blade spot found

        x1 = x0 + iw; y1 = y0 + ih
        # Soft alpha from instance mask: nonzero where class_id present
        bin_msk = (ins_msk > 0).astype(np.float32)
        if self.feather > 0:
            k = max(1, 2 * self.feather + 1)
            alpha = cv2.GaussianBlur(bin_msk, (k, k), self.feather)
        else:
            alpha = bin_msk
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha3 = alpha[..., None]

        roi_img = img[y0:y1, x0:x1].astype(np.float32)
        composed = (ins_img.astype(np.float32) * alpha3 + roi_img * (1.0 - alpha3))
        img[y0:y1, x0:x1] = np.clip(composed, 0, 255).astype(np.uint8)

        # GT: replace pixels where bin_msk == 1 with entry's class_id
        roi_gt = gt[y0:y1, x0:x1]
        # Only overwrite if pixel was BG or rare class — protect majority class regions
        ovr = (bin_msk > 0.5) & (roi_gt == 0)
        roi_gt[ovr] = entry["class_id"]
        gt[y0:y1, x0:x1] = roi_gt

    def transform(self, results):
        if self._total == 0:
            return results
        if self.rng.random() > self.prob:
            return results
        img = results.get("img")
        gt = results.get("gt_seg_map")
        if img is None or gt is None:
            return results
        H, W = gt.shape
        blade_mask = self._load_blade_mask(results.get("img_path", ""), H, W)

        n = self.rng.randint(*self.n_pastes_range)
        # Pick weighted toward classes with fewer existing pixels (rare boost)
        present = set(np.unique(gt).tolist())
        weights = []
        cls_list = []
        for c in self.target_classes:
            if not self.by_class[c]:
                continue
            cls_list.append(c)
            weights.append(2.0 if c not in present else 1.0)
        if not cls_list:
            return results

        for _ in range(n):
            c = self.rng.choices(cls_list, weights=weights, k=1)[0]
            entry = self.rng.choice(self.by_class[c])
            try:
                self._paste_one(img, gt, blade_mask, entry)
            except Exception:
                continue

        results["img"] = img
        results["gt_seg_map"] = gt
        return results
