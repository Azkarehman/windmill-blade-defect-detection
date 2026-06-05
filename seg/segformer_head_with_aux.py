"""SegformerHead variant with an auxiliary tile-level binary classification head.

Same as the stock SegformerHead but the forward also returns a per-image (per-tile
since each forward processes one 1008x1008 patch) "is this tile defect?" logit
computed from globally-pooled fused features.

Training:
  forward returns (seg_logits, aux_logit)
  loss_by_feat computes existing seg losses + BCE on aux_logit vs
  (any defect pixel in GT mask).

Inference:
  predict_by_feat returns just the seg_logits (the wrapper script reads the
  aux logit via a hook for tile gating).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.models.decode_heads.segformer_head import SegformerHead
from mmseg.registry import MODELS
from typing import List, Tuple, Optional


@MODELS.register_module()
class SegformerHeadWithAux(SegformerHead):
    """SegformerHead + aux tile-level binary classification branch."""

    def __init__(self,
                 aux_loss_weight: float = 0.1,
                 aux_hidden_channels: int = 128,
                 ignore_index: int = 255,
                 **kwargs):
        super().__init__(**kwargs)
        self.aux_loss_weight = aux_loss_weight
        self._ignore_index = ignore_index

        # Aux branch: global avg pool of fused features → small MLP → 1 logit
        self.aux_pool = nn.AdaptiveAvgPool2d(1)
        self.aux_fc1 = nn.Linear(self.channels, aux_hidden_channels)
        self.aux_fc2 = nn.Linear(aux_hidden_channels, 1)
        self.aux_act = nn.GELU()
        self.aux_bce = nn.BCEWithLogitsLoss(reduction='mean')

        # Storage for the latest aux logit (so eval scripts can read it via hook).
        # NOTE: in DDP this is per-process; do not rely on it being the most recent
        # value if you have async ops.
        self._last_aux_logit = None

    def _compute_aux_logit(self, fused: torch.Tensor) -> torch.Tensor:
        # fused: (B, C, H, W)
        pooled = self.aux_pool(fused).flatten(1)             # (B, C)
        h = self.aux_act(self.aux_fc1(pooled))               # (B, aux_hidden)
        logit = self.aux_fc2(h).squeeze(-1)                  # (B,)
        return logit

    def forward(self, inputs):
        # Same as parent forward but also computes aux logit.
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            from mmseg.models.utils import resize
            outs.append(resize(input=conv(x), size=inputs[0].shape[2:],
                               mode=self.interpolate_mode,
                               align_corners=self.align_corners))
        fused = self.fusion_conv(torch.cat(outs, dim=1))     # (B, C, H, W)
        # Aux logit from fused features
        aux_logit = self._compute_aux_logit(fused)
        self._last_aux_logit = aux_logit.detach()
        # Seg logits
        seg_logits = self.cls_seg(fused)                     # (B, num_classes, H, W)
        # Return a tuple; loss_by_feat unpacks
        return (seg_logits, aux_logit)

    # ---- override loss path to consume the tuple ----
    def loss_by_feat(self, seg_logits, batch_data_samples):
        # seg_logits is (seg, aux). Extract.
        if isinstance(seg_logits, (tuple, list)):
            seg_pred, aux_logit = seg_logits
        else:
            # safety fallback
            seg_pred = seg_logits; aux_logit = None
        # 1) usual seg losses
        loss_dict = super().loss_by_feat(seg_pred, batch_data_samples)
        # 2) aux BCE loss
        if aux_logit is not None and self.aux_loss_weight > 0:
            # Build aux targets from batch GT masks. Each sample's mask is in data_sample.gt_sem_seg.data
            # shape (1, H, W) with values 0..num_classes-1 OR 255 ignore
            B = aux_logit.shape[0]
            targets = []
            for ds in batch_data_samples:
                gt = ds.gt_sem_seg.data            # (1, H, W) tensor
                # Defect = any class in [1, num_classes-1]
                has_defect = ((gt >= 1) & (gt <= self.num_classes - 1)).any()
                targets.append(has_defect)
            target_t = torch.stack(targets).float().to(aux_logit.device)
            aux_loss = self.aux_bce(aux_logit, target_t) * self.aux_loss_weight
            loss_dict['loss_aux_cls'] = aux_loss
            # Also expose the aux head's positive rate for monitoring
            with torch.no_grad():
                aux_prob = aux_logit.sigmoid()
                loss_dict['aux_pred_pos_rate'] = aux_prob.mean()
                loss_dict['aux_gt_pos_rate']   = target_t.mean()
        return loss_dict

    def predict_by_feat(self, seg_logits, batch_img_metas):
        # During inference, extract just the seg_logits and discard aux.
        if isinstance(seg_logits, (tuple, list)):
            seg_pred, aux_logit = seg_logits
            # Stash on self so external hooks can read it
            self._last_aux_logit = aux_logit.detach()
        else:
            seg_pred = seg_logits
        return super().predict_by_feat(seg_pred, batch_img_metas)
