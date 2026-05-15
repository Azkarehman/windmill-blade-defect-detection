"""DeepLabV3+ with ResNet-152 backbone on the team's data pipeline, 5-class.

Goal: a CNN baseline for the SAM-LoRA experiment. Same transforms, same loss,
same optimizer shape. Different model + 5-class merged output (BG + 4 defects).

What's the same as team / Sam3 run:
  - WindBlade source data + WTG-disjoint splits
  - BladeFilteredRandomCrop (full pipeline) + RareClassCrop (rare)
  - RandomResize(0.5-2.0) + RandomFlip + PhotoMetricDistortion
  - ConcatDataset(full + RepeatDataset×5(rare))
  - AdamW lr=6e-5, wd=0.01, decode_head lr_mult=10×
  - LinearLR warmup → PolyLR
  - CrossEntropyLoss
  - sliding-window test mode

What's different:
  - Model: ResNet-152 + DeepLabV3+ (vs SegFormer-B4 / SAM 3.1)
  - num_classes: 5 (merged BG + 4 defects, vs team's 9)
  - Dataset: WindBladeMergedDataset (applies 9→5 LUT at load) + ApplyMergedLUT
  - rare_classes: [2, 3, 4] in merged space (vs team's [3..8] in 9-class)
  - crop_size: 1024×1024 (no SAM RoPE constraint here)
  - All params trainable (ImageNet pretrained init, like the team)

Usage:
    cd mmsegmentation
    export PYTHONPATH=~/workspace/jongwon/dmdd_pipeline/seg:$PYTHONPATH
    python tools/train.py \\
        ~/workspace/jongwon/dmdd_pipeline/seg/configs/deeplabv3plus_r152_windblade_5class.py
"""

_base_ = [
    '../../../mmsegmentation/configs/_base_/models/deeplabv3plus_r50-d8.py',
    '../../../mmsegmentation/configs/_base_/default_runtime.py',
    '../../../mmsegmentation/configs/_base_/schedules/schedule_160k.py',
]

# Make WindBladeMergedDataset + ApplyMergedLUT visible
custom_imports = dict(
    imports=['windblade_merged'],
    allow_failed_imports=False,
)

crop_size = (1024, 1024)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)

# ---- Model -------------------------------------------------------------------
# Override only what differs from deeplabv3plus_r50-d8 base. Pretrained URL
# follows the same convention as r50-d8 (open-mmlab:// scheme).
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='mmcls://resnet152',
    backbone=dict(type='ResNet', depth=152),
    decode_head=dict(num_classes=5),
    auxiliary_head=dict(num_classes=5),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(512, 512)),
)

# ---- Optimizer / schedule ----------------------------------------------------
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'decode_head': dict(lr_mult=10.),
            'auxiliary_head': dict(lr_mult=10.),
        },
    ),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=500, end=30000, by_epoch=False),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=30000, val_interval=30001)
val_cfg = None
val_dataloader = None
val_evaluator = None
test_cfg = dict(type='TestLoop')
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])

# ---- Datasets ----------------------------------------------------------------
data_root = '/home/work/workspace/jongwon/dmdd/data'
blade_mask_dir = f'{data_root}/blade_masks_640'
# Rare class IDs in MERGED 5-class space:
#   1=Laminate_Surface (common), 2=Laminate_Crack (rare),
#   3=Bond (rare), 4=Receptor (rare)
rare_class_ids = [2, 3, 4]

# Full pipeline: ApplyMergedLUT immediately after LoadAnnotations
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='ApplyMergedLUT'),
    dict(type='RandomResize', scale=(8256, 5504), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='BladeFilteredRandomCrop',
         crop_size=crop_size,
         blade_mask_dir=blade_mask_dir,
         blade_classes=(1, 2),
         min_blade_ratio=0.3,
         max_retry=50,
         center_jitter=0.3,
         cat_max_ratio=1.0),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

train_pipeline_rare = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='ApplyMergedLUT'),
    dict(type='RandomResize', scale=(8256, 5504), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RareClassCrop',
         crop_size=crop_size,
         rare_classes=rare_class_ids,
         rare_crop_prob=0.8,
         cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

train_dataset_full = dict(
    type='WindBladeMergedDataset',
    ann_file=f'{data_root}/train.json',
    merge_mode='5class',
    # Filter WTGs that appear in val or test to keep splits disjoint.
    exclude_wtg_jsons=[
        f'{data_root}/val_diuid_7.json',
        f'{data_root}/test_diuid_10.json',
    ],
    pipeline=train_pipeline,
)

train_dataset_rare = dict(
    type='RepeatDataset',
    times=5,
    dataset=dict(
        type='WindBladeMergedDataset',
        ann_file=f'{data_root}/train_rare.json',
        merge_mode='5class',
        exclude_wtg_jsons=[
            f'{data_root}/val_diuid_7.json',
            f'{data_root}/test_diuid_10.json',
        ],
        pipeline=train_pipeline_rare,
    ),
)

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[train_dataset_full, train_dataset_rare],
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type='WindBladeMergedDataset',
        ann_file=f'{data_root}/test_diuid_10.json',
        merge_mode='5class',
    ),
)

# ---- Runtime -----------------------------------------------------------------
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1, show=False),
)

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

custom_hooks = [
    dict(type='FastValHook',
         data_json=f'{data_root}/val_diuid_7.json',
         val_interval=5000,
         num_classes=5,
         initial_val=False,
         blade_filter_mask_dir=blade_mask_dir,
         blade_filter_classes=(1, 2)),
]

work_dir = '/home/work/workspace/azka5/dmdd_pipeline/runs/deeplabv3plus_r152_5class'

import os
os.environ['TZ'] = 'Asia/Seoul'
import time
time.tzset()
