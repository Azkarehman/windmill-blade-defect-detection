"""SAM 3.1 LoRA — v2 recipe + Overseas data + dilated-blade black-bg input.

Differences vs the original v2 (`sam3_lora_windblade_8class_v2.py`):

  1. Training set adds Overseas:
       train_pool = ConcatDataset(
         train.json + overseas_train.json,                          # full
         RepeatDataset×5(train_rare.json + overseas_train_rare.json) # rare
       )
     ×5 (not ×10) because the rare pool is ~2.6× larger now (2699 → 6995),
     so the effective rare-class exposure stays comparable to v2.

  2. New transform ZeroOffBladePixels:
       Loads the 640-edge blade mask, dilates with 51×51 ellipse kernel × 5
       iterations (~125 px expansion), resizes to image shape, and zeros all
       off-blade pixels. Applied right after LoadImageFromFile so all
       subsequent transforms operate on the black-bg image.

  3. blade_mask_dir = unified symlink root that contains BOTH the team's
     blade_masks_640/<wtg>/ and the overseas blade_masks_640/<task>/.

  4. work_dir = runs/sam3_lora_v2_overseas_blackbg_tight (separate from v2's dir).

Same as v2 elsewhere: class-weighted CE + Dice, LoRA r=8/α=16, neck trainable,
crop 1008², AdamW 6e-5, LinearLR→PolyLR to 30k iters, FastValHookV2 every 5k.

No `load_from` — trains from SAM 3.1's HF-pretrained weights (the Sam3Backbone
loads them at module init).
"""

_base_ = [
    '/home/work/workspace/jongwon/dmdd/mmsegmentation/configs/_base_/models/segformer_mit-b0.py',
    '/home/work/workspace/jongwon/dmdd/mmsegmentation/configs/_base_/datasets/windblade_7class_with_blade_mask.py',
    '/home/work/workspace/jongwon/dmdd/mmsegmentation/configs/_base_/default_runtime.py',
    '/home/work/workspace/jongwon/dmdd/mmsegmentation/configs/_base_/schedules/schedule_160k.py',
]

custom_imports = dict(
    imports=['_torch_compat', 'sam3_backbone', 'fast_val_hook_v2', 'zero_off_blade'],
    allow_failed_imports=False,
)

crop_size = (1008, 1008)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)

CLASS_WEIGHTS = [0.5, 1.0, 1.5, 2.5, 3.0, 3.0, 4.0, 2.0, 1.5]

# ---- Model -------------------------------------------------------------------
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type='Sam3Backbone',
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        freeze_trunk_other_than_lora=True,
        train_neck=True,
    ),
    decode_head=dict(
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        num_classes=9,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=CLASS_WEIGHTS,
                loss_name='loss_cwce',
            ),
            dict(
                type='DiceLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                ignore_index=255,
                eps=1e-5,
                loss_name='loss_dice',
            ),
        ],
    ),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(504, 504)),
)

# ---- Optimizer / schedule ----------------------------------------------------
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    dtype='bfloat16',
    optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block':   dict(decay_mult=0.),
            'norm':        dict(decay_mult=0.),
            'decode_head': dict(lr_mult=10.),
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

# ---- Datasets ----------------------------------------------------------------
data_root = '/home/work/workspace/jongwon/dmdd/data'
overseas_root = '/home/work/workspace/azka5/dmdd_pipeline/data/overseas'
# Unified blade-mask root contains symlinks to both existing-WTG dirs and
# overseas-task dirs. Same dir is used by both BladeFilteredRandomCrop (for
# choosing on-blade crops) and the new ZeroOffBladePixels transform.
blade_mask_dir = '/home/work/workspace/azka5/dmdd_pipeline/data/blade_masks_unified'
rare_class_ids = [3, 4, 5, 6, 7, 8]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # Heavy-dilated blade mask -> zero out off-blade pixels in the image.
    dict(type='ZeroOffBladePixels',
         blade_mask_dir=blade_mask_dir,
         blade_classes=(1, 2),
         dilate_kernel_size=21,
         dilate_iterations=2),
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
    dict(type='ZeroOffBladePixels',
         blade_mask_dir=blade_mask_dir,
         blade_classes=(1, 2),
         dilate_kernel_size=21,
         dilate_iterations=2),
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

# --- existing + overseas full pool ---
existing_train_full = dict(
    type='WindBlade7ClassDataset',
    ann_file=f'{data_root}/train.json',
    pipeline=train_pipeline,
)
overseas_train_full = dict(
    type='WindBlade7ClassDataset',
    ann_file=f'{overseas_root}/overseas_train.json',
    pipeline=train_pipeline,
)

# --- existing + overseas rare pool, ×5 repeat ---
existing_train_rare = dict(
    type='RepeatDataset',
    times=5,
    dataset=dict(
        type='WindBlade7ClassDataset',
        ann_file=f'{data_root}/train_rare.json',
        pipeline=train_pipeline_rare,
    ),
)
overseas_train_rare = dict(
    type='RepeatDataset',
    times=5,
    dataset=dict(
        type='WindBlade7ClassDataset',
        ann_file=f'{overseas_root}/overseas_train_rare.json',
        pipeline=train_pipeline_rare,
    ),
)

train_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            existing_train_full, overseas_train_full,
            existing_train_rare, overseas_train_rare,
        ],
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(ann_file=f'{data_root}/test.json'),
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
    dict(type='FastValHookV2',
         data_json=f'{data_root}/val_diuid_7.json',
         val_interval=5000,
         num_classes=9,
         initial_val=False,
         blade_filter_mask_dir=blade_mask_dir,
         blade_filter_classes=(1, 2),
         num_workers=8,
         prefetch_factor=2,
         use_bf16=True,
         persistent_workers=True,
         slide_batch_size=8),
]

work_dir = '/home/work/workspace/azka5/dmdd_pipeline/runs/sam3_lora_v2_overseas_blackbg_tight'

import os
os.environ['TZ'] = 'Asia/Seoul'
import time
time.tzset()
