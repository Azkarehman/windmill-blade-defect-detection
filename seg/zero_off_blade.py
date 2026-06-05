"""mmseg transform: ZeroOffBladePixels.

Loads the precomputed blade mask at <blade_mask_dir>/<image_parent>/<stem>.png,
dilates it heavily (default 51×51 ellipse kernel × 5 iterations ≈ ~125 px
expansion at 640-edge), resizes to image shape, and zeros out off-blade image
pixels. Applied EARLY in the pipeline (after LoadImageFromFile) so subsequent
RandomResize/Crop/PhotoMetricDistortion operate on the already-black-bg image.

Blade mask values: 0=BG, 1=blade, 2=nose, 3=pole. We dilate the union of
blade_classes (default (1, 2)) since that's the team's existing on-blade
definition for BladeFilteredRandomCrop.

If a blade mask file is missing, by default the image is left unchanged
(fallback_keep_all=True). Set False to zero the entire image when missing
(useful for strict sanity checks).
"""
import os
from pathlib import Path

import cv2
import numpy as np

from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ZeroOffBladePixels(BaseTransform):
    def __init__(
        self,
        blade_mask_dir: str,
        blade_classes=(1, 2),
        dilate_kernel_size: int = 51,
        dilate_iterations: int = 5,
        fallback_keep_all: bool = True,
    ):
        super().__init__()
        self.blade_mask_dir = blade_mask_dir
        self.blade_classes = tuple(blade_classes)
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size)
        )
        self.iterations = dilate_iterations
        self.fallback_keep_all = fallback_keep_all

    def _load_dilated_blade_mask(self, image_path: str, H: int, W: int):
        stem = Path(image_path).stem
        parent = Path(image_path).parent.name
        mp = os.path.join(self.blade_mask_dir, parent, f"{stem}.png")
        if not os.path.isfile(mp):
            return None
        bm = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        if bm is None:
            return None
        # Binary union of blade_classes
        bin_mask = np.zeros_like(bm, dtype=np.uint8)
        for c in self.blade_classes:
            bin_mask = bin_mask | (bm == c).astype(np.uint8)
        # Dilate at the mask's native resolution first (faster than resize-then-dilate)
        dilated = cv2.dilate(bin_mask, self.kernel, iterations=self.iterations)
        # Resize up to image shape with NEAREST
        if dilated.shape != (H, W):
            dilated = cv2.resize(dilated, (W, H), interpolation=cv2.INTER_NEAREST)
        return dilated

    def transform(self, results):
        img = results.get("img")
        img_path = results.get("img_path")
        if img is None or img_path is None:
            return results
        H, W = img.shape[:2]
        mask = self._load_dilated_blade_mask(img_path, H, W)
        if mask is None:
            if self.fallback_keep_all:
                return results
            # else fall through to zero (rare/strict path)
            mask = np.zeros((H, W), dtype=np.uint8)
        # Multiply image by the binary mask (broadcast across channels)
        out = img * mask[..., None]
        results["img"] = out.astype(img.dtype)
        return results
