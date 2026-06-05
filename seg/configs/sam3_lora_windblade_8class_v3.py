"""SAM 3.1 LoRA v3 — decoder-head-only fine-tune of v2 with data-centric fixes.

What changes from v2 (see EXPERIMENTS.md for v2 details):

  1. Backbone + LoRA + neck all FROZEN (paramwise lr_mult=0).
     Only the SegformerHead decode head trains.
     Rationale: v2's encoder representations are already good (per the FN
     analysis: 27 % of FNs are mislocated CCs of the right class — the encoder
     has the signal). The decoder is the part doing class disambiguation.

  2. Loss swap: CW-CE + Dice  ->  ClassPairPenaltyLoss + Dice
     ClassPairPenaltyLoss = class-weighted CE plus a softmax-prob penalty on
     the confused-pair classes per the v2 FN failure-mode analysis.

  3. Training data: original train pipeline + a new CopyPasteRareInstances
     transform that pastes rare-class instances (3,4,5,6,7) into images.
     Targets the 73 % "completely missed" FN bucket — more instances of those
     classes in training.

  4. load_from: v2's iter_30000.pth (weights, fresh schedule).

  5. max_iters: 30000 -> 8000. Decoder-only + warm start = fast convergence.

  6. work_dir: runs/sam3_lora_v3.
"""

_base_ = [
    '/home/work/workspace/jongwon/dmdd/mmsegmentation/configs/_base_/models/segformer_mit-b0.py',
    '/home/work/workspace/jongwon/dmdd/mmsegmentation/configs/_base_/datasets/windblade_7class_with_blade_mask.py',
    '/home/work/workspace/jongwon/dmdd/mmsegmentation/configs/_base_/default_runtime.py',
    '/home/work/workspace/jongwon/dmdd/mmsegmentation/configs/_base_/schedules/schedule_160k.py',
]

custom_imports = dict(
    imports=[
        '_torch_compat',
        'sam3_backbone',
        'fast_val_hook_v2',
        'copy_paste_aug',
        'class_pair_penalty_loss',
    ],
    allow_failed_imports=False,
)

load_from = '/home/work/workspace/azka5/dmdd_pipeline/runs/sam3_lora_v2/iter_30000.pth'

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
CONFUSED_PAIRS = {
    1: [2], 2: [1],
    3: [5, 6], 5: [3], 6: [3],
    7: [8], 8: [7],
}

# ---- Model -------------------------------------------------------------------
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type='Sam3Backbone',
        lora_r=8, lora_alpha=16, lora_dropout=0.1,
        freeze_trunk_other_than_lora=True,
        train_neck=True,                   # neck params will be lr=0 below
    ),
    decode_head=dict(
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        num_classes=9,
        loss_decode=[
            dict(
                type='ClassPairPenaltyLoss',
                num_classes=9,
                class_weight=CLASS_WEIGHTS,
                confused_pairs=CONFUSED_PAIRS,
                penalty_lambda=0.3,
                loss_weight=1.0,
                loss_name='loss_cppen',
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
# Freeze backbone + neck via lr_mult=0; only decode_head trains.
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    dtype='bfloat16',
    optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'backbone':    dict(lr_mult=0.0, decay_mult=0.0),
            'decode_head': dict(lr_mult=10.),
            'pos_block':   dict(decay_mult=0.),
            'norm':        dict(decay_mult=0.),
        },
    ),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=200),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=200, end=8000, by_epoch=False),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=8000, val_interval=8001)
val_cfg = None
val_dataloader = None
val_evaluator = None
test_cfg = dict(type='TestLoop')

# ---- Datasets ----------------------------------------------------------------
data_root = '/home/work/workspace/jongwon/dmdd/data'
blade_mask_dir = f'{data_root}/blade_masks_640'
rare_class_ids = [3, 4, 5, 6, 7, 8]
pool_dir = '/home/work/workspace/azka5/dmdd_pipeline/data/instance_pool'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(8256, 5504), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='BladeFilteredRandomCrop',
         crop_size=crop_size, blade_mask_dir=blade_mask_dir,
         blade_classes=(1, 2), min_blade_ratio=0.3, max_retry=50,
         center_jitter=0.3, cat_max_ratio=1.0),
    dict(type='CopyPasteRareInstances',
         pool_dir=pool_dir, blade_mask_dir=blade_mask_dir,
         target_classes=[3, 4, 5, 6, 7], prob=0.7,
         n_pastes_range=(1, 3), max_paste_frac=0.10, feather=5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

train_pipeline_rare = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(8256, 5504), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RareClassCrop',
         crop_size=crop_size, rare_classes=rare_class_ids,
         rare_crop_prob=0.8, cat_max_ratio=0.75),
    dict(type='CopyPasteRareInstances',
         pool_dir=pool_dir, blade_mask_dir=blade_mask_dir,
         target_classes=[3, 4, 5, 6, 7], prob=0.5,
         n_pastes_range=(1, 2), max_paste_frac=0.10, feather=5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

train_dataset_full = dict(
    type='WindBlade7ClassDataset',
    ann_file=f'{data_root}/train.json',
    pipeline=train_pipeline,
)

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
    batch_size=1, num_workers=4, persistent_workers=True,
    dataset=dict(ann_file=f'{data_root}/test.json'),
)

# ---- Runtime -----------------------------------------------------------------
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1, show=False),
)

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

custom_hooks = [
    dict(type='FastValHookV2',
         data_json=f'{data_root}/val_diuid_7.json',
         val_interval=2000,
         num_classes=9,
         initial_val=False,
         blade_filter_mask_dir=blade_mask_dir,
         blade_filter_classes=(1, 2),
         num_workers=8, prefetch_factor=2,
         use_bf16=True, persistent_workers=True,
         slide_batch_size=8),
]

work_dir = '/home/work/workspace/azka5/dmdd_pipeline/runs/sam3_lora_v3'

import os
os.environ['TZ'] = 'Asia/Seoul'
import time
time.tzset()
