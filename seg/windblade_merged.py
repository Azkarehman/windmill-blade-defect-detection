"""
mmseg dataset wrapper that LUT-remaps 9-class defect masks to the merged
taxonomy at load time. Drop in to your mmseg install via the registry.

5-class merged:
    0 = Background
    1 = Laminate Surface  (orig 1, 2)
    2 = Laminate Crack    (orig 3, 4)
    3 = Bond              (orig 5, 6)  - trailing_edge only (filename rule)
    4 = Receptor          (orig 7, 8)

For BINARY (FG/BG), set merge_mode='binary' on the config — collapses everything
defect to 1.
"""
import json
import os
from typing import List

import numpy as np

try:
    from mmseg.datasets.basesegdataset import BaseSegDataset
    from mmseg.registry import DATASETS
except ImportError:
    # Allow the file to be importable outside the mmseg env for syntax checks
    BaseSegDataset = object
    def DATASETS(cls): return cls  # noqa


# Merged 5-class LUT (defect 1-8 → 1-4)
LUT_5CLASS = np.zeros(256, dtype=np.uint8)
LUT_5CLASS[1] = 1; LUT_5CLASS[2] = 1
LUT_5CLASS[3] = 2; LUT_5CLASS[4] = 2
LUT_5CLASS[5] = 3; LUT_5CLASS[6] = 3
LUT_5CLASS[7] = 4; LUT_5CLASS[8] = 4
LUT_5CLASS[255] = 255

# Binary LUT (any defect → 1)
LUT_BINARY = np.zeros(256, dtype=np.uint8)
LUT_BINARY[1:9] = 1
LUT_BINARY[255] = 255


@DATASETS.register_module()
class WindBladeMergedDataset(BaseSegDataset):
    """Merged 5-class or binary defect segmentation dataset.

    Reads {image_path, mask_path} from a JSON file (same format used by
    train.json / train_rare.json / val_full.json / etc).

    Args:
        ann_file: path to JSON annotation list
        merge_mode: '5class' (default) or 'binary'
    """

    METAINFO_5CLASS = dict(
        classes=("background", "laminate_surface", "laminate_crack",
                 "bond", "receptor"),
        palette=[[0, 0, 0], [255, 255, 0], [0, 200, 0],
                 [0, 255, 255], [255, 0, 128]],
    )
    METAINFO_BINARY = dict(
        classes=("background", "defect"),
        palette=[[0, 0, 0], [255, 0, 0]],
    )

    def __init__(self, ann_file, merge_mode: str = "5class",
                 exclude_wtg_jsons=None, **kwargs):
        """
        Args:
            ann_file: split JSON to load
            merge_mode: '5class' or 'binary' (LUT applied at load time)
            exclude_wtg_jsons: iterable of paths to OTHER split JSONs (val/test).
                Any WTG appearing in those JSONs is filtered out of this split.
                Pass this on the TRAIN dataset to prevent per-WTG leakage with val
                and test. Pass None (default) for the val/test datasets themselves.
        """
        assert merge_mode in ("5class", "binary"), merge_mode
        self.merge_mode = merge_mode
        self._lut = LUT_5CLASS if merge_mode == "5class" else LUT_BINARY
        self._exclude_wtg_jsons = list(exclude_wtg_jsons) if exclude_wtg_jsons else []
        # Set METAINFO before super().__init__ uses it
        self.METAINFO = (self.METAINFO_5CLASS if merge_mode == "5class"
                         else self.METAINFO_BINARY)
        super().__init__(
            ann_file=ann_file,
            data_root="",
            data_prefix=dict(img_path="", seg_map_path=""),
            reduce_zero_label=False,
            **kwargs)

    @staticmethod
    def _wtg(image_path: str) -> str:
        return os.path.basename(os.path.dirname(image_path))

    def _build_excluded_set(self) -> set:
        excluded = set()
        for p in self._exclude_wtg_jsons:
            if not p or not os.path.exists(p):
                continue
            with open(p) as f:
                for e in json.load(f):
                    excluded.add(self._wtg(e["image_path"]))
        return excluded

    def load_data_list(self) -> List[dict]:
        data_list = []
        skipped_missing = 0
        skipped_leak = 0
        with open(self.ann_file) as f:
            entries = json.load(f)
        excluded_wtgs = self._build_excluded_set()
        for ann in entries:
            img_path = ann["image_path"]
            msk_path = ann["mask_path"]
            if excluded_wtgs and self._wtg(img_path) in excluded_wtgs:
                skipped_leak += 1
                continue
            if not (os.path.exists(img_path) and os.path.exists(msk_path)):
                skipped_missing += 1
                continue
            data_list.append(dict(
                img_path=img_path,
                seg_map_path=msk_path,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                seg_fields=[],
                _merged_lut=self._lut,
            ))
        msg = (f"WindBladeMergedDataset[{os.path.basename(self.ann_file)}]: "
               f"kept={len(data_list)} skipped_missing={skipped_missing} "
               f"skipped_leak={skipped_leak} (held-out WTGs)")
        print(msg)
        return data_list


# ----------------------------------------------------------------------------
# Pipeline transform that applies the LUT to gt_seg_map after LoadAnnotations.
# Register it and place IMMEDIATELY AFTER `LoadAnnotations` in the pipeline.
# ----------------------------------------------------------------------------
try:
    from mmcv.transforms.base import BaseTransform
    from mmseg.registry import TRANSFORMS
except ImportError:
    BaseTransform = object
    def TRANSFORMS(cls): return cls  # noqa


@TRANSFORMS.register_module()
class ApplyMergedLUT(BaseTransform):
    """Apply the merged-class LUT to gt_seg_map (and any extra seg fields).

    The LUT is attached to each data sample by WindBladeMergedDataset.
    """

    def transform(self, results: dict) -> dict:
        lut = results.get("_merged_lut")
        if lut is None:
            return results
        for key in results.get("seg_fields", ["gt_seg_map"]):
            if key in results:
                results[key] = lut[results[key]]
        return results
