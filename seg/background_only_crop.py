"""BackgroundOnlyCrop — reject-sample crops that contain ZERO defect pixels.

Used to generate "non-defect" patches for v8's 1-defect + 2-background sampling
strategy. Tries up to max_retry random crops; if none are pure-BG, falls back
to the crop with the smallest defect-pixel count.

Defect = any class id != BG (0) and != ignore_index (255). Treats class 0 and
255 as non-defect.
"""
import numpy as np
import mmcv
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class BackgroundOnlyCrop(BaseTransform):
    def __init__(
        self,
        crop_size,
        max_retry: int = 30,
        ignore_index: int = 255,
        bg_class: int = 0,
        max_defect_pixel_ratio: float = 0.0,
    ):
        self.crop_size = tuple(crop_size) if isinstance(crop_size, (list, tuple)) else (crop_size, crop_size)
        self.max_retry = max_retry
        self.ignore_index = ignore_index
        self.bg_class = bg_class
        self.max_defect_pixel_ratio = max_defect_pixel_ratio

    def _generate_random_crop_bbox(self, h, w):
        crop_h, crop_w = self.crop_size
        y0 = np.random.randint(0, max(1, h - crop_h + 1))
        x0 = np.random.randint(0, max(1, w - crop_w + 1))
        return y0, x0, y0 + crop_h, x0 + crop_w

    def _defect_ratio(self, seg_crop):
        # Non-BG, non-ignore pixels are defects.
        mask = (seg_crop != self.bg_class) & (seg_crop != self.ignore_index)
        return float(mask.sum()) / max(1, mask.size)

    @staticmethod
    def _crop(arr, bbox):
        y0, x0, y1, x1 = bbox
        return arr[y0:y1, x0:x1] if arr.ndim == 2 else arr[y0:y1, x0:x1, :]

    def transform(self, results: dict) -> dict:
        img = results['img']
        seg_map = results.get('gt_seg_map')
        if seg_map is None:
            # Fall back to plain random crop if no seg
            h, w = img.shape[:2]
            bbox = self._generate_random_crop_bbox(h, w)
            results['img'] = self._crop(img, bbox)
            return results

        h, w = seg_map.shape[:2]
        best_bbox, best_ratio = None, 1.0
        for _ in range(self.max_retry):
            bbox = self._generate_random_crop_bbox(h, w)
            crop_seg = self._crop(seg_map, bbox)
            r = self._defect_ratio(crop_seg)
            if r <= self.max_defect_pixel_ratio:
                best_bbox, best_ratio = bbox, r
                break
            if r < best_ratio:
                best_bbox, best_ratio = bbox, r

        if best_bbox is None:
            best_bbox = self._generate_random_crop_bbox(h, w)

        results['img'] = self._crop(img, best_bbox)
        results['gt_seg_map'] = self._crop(seg_map, best_bbox)
        results['img_shape'] = results['img'].shape[:2]
        if 'pad_shape' in results:
            results['pad_shape'] = results['img'].shape
        return results

    def __repr__(self):
        return (f'BackgroundOnlyCrop(crop_size={self.crop_size}, '
                f'max_retry={self.max_retry}, '
                f'max_defect_pixel_ratio={self.max_defect_pixel_ratio})')
