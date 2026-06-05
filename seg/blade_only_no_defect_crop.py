"""BladeOnlyNoDefectCrop — reject-sample crops that are on the blade BUT contain
ZERO defect pixels. The hard-negative companion to defect-centered crops.

Two filters combined:
  1. Blade ratio (using blade_mask_dir + image stem) >= min_blade_ratio.
     Avoids trivial off-blade crops (sky, hub, ground) — those teach the model
     nothing useful since "no defect on sky" is not the hard case.
  2. Defect ratio (in gt_seg_map) == 0 (or below max_defect_pixel_ratio).
     Crop must contain no defect annotations of any class.

If max_retry random crops fail to satisfy both, returns the crop with the
lowest defect_ratio that meets the blade requirement, or pure-random fallback.
"""
import os.path as osp

import numpy as np
import cv2
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class BladeOnlyNoDefectCrop(BaseTransform):
    def __init__(
        self,
        crop_size,
        blade_mask_dir: str,
        blade_classes=(1, 2),
        min_blade_ratio: float = 0.3,
        max_retry: int = 50,
        ignore_index: int = 255,
        bg_class: int = 0,
        max_defect_pixel_ratio: float = 0.0,
    ):
        self.crop_size = tuple(crop_size) if isinstance(crop_size, (list, tuple)) else (crop_size, crop_size)
        self.blade_mask_dir = blade_mask_dir
        self.blade_classes = tuple(blade_classes)
        self.min_blade_ratio = min_blade_ratio
        self.max_retry = max_retry
        self.ignore_index = ignore_index
        self.bg_class = bg_class
        self.max_defect_pixel_ratio = max_defect_pixel_ratio

    def _blade_mask_path(self, img_path):
        wtg = osp.basename(osp.dirname(img_path))
        stem = osp.splitext(osp.basename(img_path))[0]
        return osp.join(self.blade_mask_dir, wtg, stem + '.png')

    def _load_blade_mask(self, img_path):
        p = self._blade_mask_path(img_path)
        if not osp.exists(p): return None
        m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if m is None: return None
        return np.isin(m, self.blade_classes).astype(np.uint8)

    def _random_bbox(self, h, w):
        ch, cw = self.crop_size
        y = np.random.randint(0, max(1, h - ch + 1))
        x = np.random.randint(0, max(1, w - cw + 1))
        return y, x, y + ch, x + cw

    def _blade_ratio_native(self, blade_mask, bbox, img_h, img_w):
        """Compute blade ratio in mask's native resolution (mask may be downsampled vs image)."""
        mh, mw = blade_mask.shape
        y1, x1, y2, x2 = bbox
        my1 = max(0, int(y1 * mh / img_h))
        my2 = min(mh, int(np.ceil(y2 * mh / img_h)))
        mx1 = max(0, int(x1 * mw / img_w))
        mx2 = min(mw, int(np.ceil(x2 * mw / img_w)))
        if my2 <= my1 or mx2 <= mx1: return 0.0
        region = blade_mask[my1:my2, mx1:mx2]
        return float(region.sum()) / region.size

    def _defect_ratio(self, seg_crop):
        mask = (seg_crop != self.bg_class) & (seg_crop != self.ignore_index)
        return float(mask.sum()) / max(1, mask.size)

    @staticmethod
    def _crop(arr, bbox):
        y1, x1, y2, x2 = bbox
        return arr[y1:y2, x1:x2] if arr.ndim == 2 else arr[y1:y2, x1:x2, :]

    def transform(self, results):
        img = results['img']
        seg_map = results.get('gt_seg_map')
        img_path = results.get('img_path') or results.get('image_path')
        h, w = img.shape[:2]

        blade_mask = self._load_blade_mask(img_path) if img_path else None

        best_bbox, best_defect_ratio = None, 1.0
        for _ in range(self.max_retry):
            bbox = self._random_bbox(h, w)
            # Blade requirement
            if blade_mask is not None:
                br = self._blade_ratio_native(blade_mask, bbox, h, w)
                if br < self.min_blade_ratio:
                    continue
            # Defect requirement
            if seg_map is not None:
                crop_seg = self._crop(seg_map, bbox)
                dr = self._defect_ratio(crop_seg)
                if dr <= self.max_defect_pixel_ratio:
                    best_bbox, best_defect_ratio = bbox, dr
                    break
                if dr < best_defect_ratio:
                    best_bbox, best_defect_ratio = bbox, dr
            else:
                # No seg map: just take any blade crop
                best_bbox = bbox; break

        if best_bbox is None:
            best_bbox = self._random_bbox(h, w)

        results['img'] = self._crop(img, best_bbox)
        if seg_map is not None:
            results['gt_seg_map'] = self._crop(seg_map, best_bbox)
        results['img_shape'] = results['img'].shape[:2]
        if 'pad_shape' in results:
            results['pad_shape'] = results['img'].shape
        return results

    def __repr__(self):
        return (f'BladeOnlyNoDefectCrop(crop_size={self.crop_size}, '
                f'min_blade_ratio={self.min_blade_ratio}, '
                f'max_defect_pixel_ratio={self.max_defect_pixel_ratio}, '
                f'max_retry={self.max_retry})')
