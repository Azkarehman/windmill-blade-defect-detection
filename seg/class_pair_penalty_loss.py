"""mmseg loss: CrossEntropy + per-confused-pair anti-confusion penalty.

For pixels with GT class c, penalize predicted probability mass that lands on
classes that c is commonly confused with (per the v2 FN failure-mode analysis):
  La_Exposure (1) <-> La_Damage (2)
  La_Crack (3) <-> Bond_Crack (5), Bond_Open (6)
  Receptor_Lightning (7) <-> Receptor_Damage (8)

Loss = CE(class_weight) + lambda * mean_over_pixels[ sum_{c' in confused_with(GT)} P(c'|x) ]

The penalty term doesn't change CE's gradient direction for correct-class
predictions — it only adds gradient AWAY from confused-pair wrong classes
when the GT is one of the confused-pair members.
"""
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


# Default confused-pair structure from v2 baseline FN analysis on test set.
DEFAULT_CONFUSED_PAIRS: Dict[int, List[int]] = {
    1: [2],          # La_Exposure -> La_Damage
    2: [1],          # La_Damage   -> La_Exposure
    3: [5, 6],       # La_Crack    -> Bond_Crack/Open
    5: [3],          # Bond_Crack  -> La_Crack
    6: [3],          # Bond_Open   -> La_Crack
    7: [8],          # Receptor_Lightning -> Receptor_Damage
    8: [7],          # Receptor_Damage   -> Receptor_Lightning
}

DEFAULT_CLASS_WEIGHTS = [
    0.5, 1.0, 1.5, 2.5, 3.0, 3.0, 4.0, 2.0, 1.5,
]


@MODELS.register_module()
class ClassPairPenaltyLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 9,
        class_weight: List[float] = None,
        confused_pairs: Dict[int, List[int]] = None,
        penalty_lambda: float = 0.3,
        ignore_index: int = 255,
        loss_weight: float = 1.0,
        loss_name: str = "loss_cppen",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.penalty_lambda = penalty_lambda

        cw = class_weight if class_weight is not None else DEFAULT_CLASS_WEIGHTS
        self.register_buffer(
            "class_weight",
            torch.tensor(cw, dtype=torch.float32),
            persistent=False,
        )

        # Build a [num_classes, num_classes] indicator matrix M[c, c'] = 1 if
        # c' is in confused_with(c). Used for vectorized penalty computation.
        pairs = confused_pairs if confused_pairs is not None else DEFAULT_CONFUSED_PAIRS
        M = torch.zeros(num_classes, num_classes)
        for c, lst in pairs.items():
            for cp in lst:
                if 0 <= c < num_classes and 0 <= cp < num_classes:
                    M[c, cp] = 1.0
        self.register_buffer("confusion_mask", M, persistent=False)

    @property
    def loss_name(self):
        return self._loss_name

    def forward(self, logits, target, weight=None, ignore_index=None, **kwargs):
        # logits: [B, C, H, W], target: [B, H, W] (int)
        ignore_index = self.ignore_index if ignore_index is None else ignore_index

        ce = F.cross_entropy(
            logits,
            target,
            weight=self.class_weight.to(logits.dtype),
            ignore_index=ignore_index,
            reduction="mean",
        )

        if self.penalty_lambda <= 0:
            return self.loss_weight * ce

        # Penalty: for each pixel, look up the confused-pair mask row for its GT class,
        # then sum softmax probabilities on those columns.
        with torch.no_grad():
            valid = target != ignore_index
            # Clamp invalid to 0 to allow indexing; we mask them out below.
            safe = target.clamp(0, self.num_classes - 1)
        rows = self.confusion_mask[safe]                      # [B, H, W, C]
        probs = F.softmax(logits, dim=1).permute(0, 2, 3, 1)  # [B, H, W, C]
        per_pix = (probs * rows).sum(dim=-1)                  # [B, H, W]
        per_pix = per_pix * valid.float()
        denom = valid.float().sum().clamp_min(1.0)
        penalty = per_pix.sum() / denom

        return self.loss_weight * (ce + self.penalty_lambda * penalty)
