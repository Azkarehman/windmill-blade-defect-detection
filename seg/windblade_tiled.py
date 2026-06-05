"""WindBladeTiledDataset — reads a tile manifest (one entry per pre-selected
tile from `precompute_v8_tile_manifest.py`) and preserves the per-entry
`tile_y`, `tile_x`, `tile_h`, `tile_w` fields so a downstream `FixedTileCrop`
transform can crop at the exact location.

Each entry in the JSON manifest should look like:
    {
      "image_path": "...",
      "mask_path":  "...",
      "tile_y":     1234,
      "tile_x":     5678,
      "tile_h":     1008,
      "tile_w":     1008,
      "kind":       "defect" | "bg",
      ... (other fields are preserved for inspection but not required)
    }
"""
import json
import os
from typing import List

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class WindBladeTiledDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'la_exposure', 'la_damage', 'la_crack', 'la_open',
                 'bond_crack', 'bond_open', 'receptor_lightning', 'receptor_damage'),
        palette=[[0,0,0],[255,128,0],[0,255,255],[255,0,255],[0,128,255],
                 [128,255,0],[0,0,255],[255,0,0],[0,255,128]],
    )

    def __init__(self, ann_file: str, **kwargs):
        super().__init__(
            ann_file=ann_file,
            data_root='',
            data_prefix=dict(img_path='', seg_map_path=''),
            reduce_zero_label=False,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        data_list = []
        skipped = 0
        with open(self.ann_file, 'r') as f:
            entries = json.load(f)
        for ann in entries:
            ip = ann['image_path']; mp = ann['mask_path']
            if not (os.path.exists(ip) and os.path.exists(mp)):
                skipped += 1; continue
            data_list.append(dict(
                img_path=ip,
                seg_map_path=mp,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                seg_fields=[],
                tile_y=int(ann['tile_y']),
                tile_x=int(ann['tile_x']),
                tile_h=int(ann.get('tile_h', 1008)),
                tile_w=int(ann.get('tile_w', 1008)),
                kind=ann.get('kind', 'unknown'),
            ))
        if skipped > 0:
            print(f'WindBladeTiledDataset: skipped {skipped} entries with missing files')
        return data_list
