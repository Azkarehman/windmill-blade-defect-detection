"""Asymmetric Loss (Ben-Baruch et al., ICCV 2021) for multi-class
semantic segmentation.

Standard ASL is defined for multi-label binary classification. For our
multi-class softmax segmentation we apply it per-pixel as one-vs-rest:
for each class c, treat softmax(logits)[:, c] as the binary probability
that the pixel belongs to class c, and use the asymmetric focal +
probability-shift terms from the paper. Sum over classes, mean over
valid pixels.

  L_pos = (1 - p)^gamma_pos     * log(p)
  L_neg = max(p - m, 0)^gamma_neg * log(1 - max(p - m, 0))
  loss  = -[ y*L_pos + (1-y)*L_neg ]  summed over C, mean over HxW

gamma_neg > gamma_pos focuses gradient on hard negatives (false positives).
The probability shift m zeros loss for confident-correct negatives.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


@MODELS.register_module()
class AsymmetricLoss(nn.Module):
    def __init__(self,
                 gamma_pos: float = 0.0,
                 gamma_neg: float = 4.0,
                 m: float = 0.05,
                 eps: float = 1e-8,
                 loss_weight: float = 1.0,
                 ignore_index: int = 255,
                 loss_name: str = 'loss_asl'):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.m = m
        self.eps = eps
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self, cls_logit, label, **kwargs):
        # cls_logit: [B, C, H, W] ; label: [B, H, W]
        B, C, H, W = cls_logit.shape
        probs = F.softmax(cls_logit, dim=1)

        valid = label != self.ignore_index
        label_safe = label.clone()
        label_safe[~valid] = 0
        target = F.one_hot(label_safe, num_classes=C).permute(0, 3, 1, 2).float()

        probs_neg_shifted = torch.clamp(probs - self.m, min=0.0)

        log_p_pos = torch.log(probs.clamp(min=self.eps))
        log_one_minus_p_neg = torch.log((1.0 - probs_neg_shifted).clamp(min=self.eps))

        loss_pos = torch.pow(1.0 - probs, self.gamma_pos) * log_p_pos
        loss_neg = torch.pow(probs_neg_shifted, self.gamma_neg) * log_one_minus_p_neg

        per_class = target * loss_pos + (1.0 - target) * loss_neg
        per_pixel = per_class.sum(dim=1)
        loss = -per_pixel[valid].mean() if valid.any() else cls_logit.sum() * 0.0
        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self._loss_name
