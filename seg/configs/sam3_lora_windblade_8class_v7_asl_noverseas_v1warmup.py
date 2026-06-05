"""SAM 3.1 LoRA, v7 — ASL + Dice on team-only data, warm-start from v1.

Ablation of v6: same recipe, but **without** overseas data. Tests whether
overseas helped or hurt v6's results. Everything else matches v6.

Design:
  - Loss: AsymmetricLoss (γ_pos=0, γ_neg=4, m=0.05) + Dice, uniform weights.
  - Backbone: Sam3Backbone (LoRA only, no encoder unfreeze).
  - Data: team train.json + RepeatDataset×10(team train_rare.json). NO overseas.
  - **No black-bg transform** — input domain matches v1's raw-image training.
  - Init: warm-start from v1 iter_30000.pth (the 64% micro_P precision baseline).
"""

_base_ = [
    '/home/work/workspace/jongwon/dmdd/mmsegmentation/configs/_base_/models/segformer_mit-b0.py',
    '/home/work/workspace/jongwon/dmdd/mmsegmentation/configs/_base_/datasets/windblade_7class_with_blade_mask.py',
    '/home/work/workspace/jongwon/dmdd/mmsegmentation/configs/_base_/default_runtime.py',
    '/home/work/workspace/jongwon/dmdd/mmsegmentation/configs/_base_/schedules/schedule_160k.py',
]

custom_imports = dict(
    imports=['_torch_compat', 'sam3_backbone', 'fast_val_hook_v2', 'asl_loss'],
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
                type='AsymmetricLoss',
                gamma_pos=0.0,
                gamma_neg=4.0,
                m=0.05,
                loss_weight=1.0,
                ignore_index=255,
                loss_name='loss_asl',
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

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    dtype='bfloat16',
    optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
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

data_root = '/home/work/workspace/jongwon/dmdd/data'
blade_mask_dir = f'{data_root}/blade_masks_640'
rare_class_ids = [3, 4, 5, 6, 7, 8]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
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
    type='WindBlade7ClassDataset',
    ann_file=f'{data_root}/train.json',
    pipeline=train_pipeline,
)

# v2-style ×10 since we no longer add overseas to the rare pool.
train_dataset_rare = dict(
    type='RepeatDataset',
    times=10,
    dataset=dict(
        type='WindBlade7ClassDataset',
        ann_file=f'{data_root}/train_rare.json',
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
        datasets=[train_dataset_full, train_dataset_rare],
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(ann_file=f'{data_root}/test.json'),
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1, show=False),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='dmdd',
            entity='usmanfamily',
            name='v7_asl_noverseas_v1warmup_iter0-30k',
            group='sam3_lora_v7_asl_noverseas_v1warmup',
            tags=['sam3.1', 'lora', 'asl-loss', 'lora-only', 'team-only', 'v1-warmup'],
            notes='Ablation of v6 minus overseas data. LoRA-only Sam3Backbone, ASL (γ_pos=0, γ_neg=4, m=0.05) + Dice, uniform class weights, team train + train_rare ×10 only, no black-bg transform, warm-start from v1 iter_30000.pth, 30k iters.',
        ),
    ),
]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

custom_hooks = [
    dict(type='FastValHookV2',
         # Full val set (2124 samples) — matches v2/v4 protocol.
         data_json=f'{data_root}/val_diuid_7.json',
         val_interval=10000,
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

work_dir = '/home/work/workspace/azka5/dmdd_pipeline/runs/sam3_lora_v7_asl_noverseas_v1warmup'

load_from = '/home/work/workspace/azka5/dmdd_pipeline/runs/sam3_lora_v1/iter_30000.pth'

import os
os.environ['TZ'] = 'Asia/Seoul'
import time
time.tzset()
