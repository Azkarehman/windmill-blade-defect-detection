"""FixedTileCrop — crops at the exact (tile_y, tile_x) carried in the
results dict by WindBladeTiledDataset. No randomness, no fallback.

Use after LoadImageFromFile + LoadAnnotations. Skip RandomResize, since tiles
are pre-defined at native image resolution. RandomFlip and PhotoMetricDistortion
can still run after this transform.
"""
import numpy as np

from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class FixedTileCrop(BaseTransform):
    def __init__(self, default_crop_size=(1008, 1008)):
        self.default_crop_size = tuple(default_crop_size)

    def transform(self, results: dict) -> dict:
        img = results['img']
        H, W = img.shape[:2]
        y = int(results.get('tile_y', 0))
        x = int(results.get('tile_x', 0))
        h = int(results.get('tile_h', self.default_crop_size[0]))
        w = int(results.get('tile_w', self.default_crop_size[1]))
        # Clamp to image bounds (handles small images defensively)
        y = max(0, min(y, max(0, H - h)))
        x = max(0, min(x, max(0, W - w)))
        results['img'] = img[y:y+h, x:x+w]
        seg = results.get('gt_seg_map')
        if seg is not None:
            results['gt_seg_map'] = seg[y:y+h, x:x+w]
        results['img_shape'] = results['img'].shape[:2]
        if 'pad_shape' in results:
            results['pad_shape'] = results['img'].shape
        return results

    def __repr__(self):
        return f'FixedTileCrop(default_crop_size={self.default_crop_size})'
