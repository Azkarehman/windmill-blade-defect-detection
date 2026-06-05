"""SAM 3.1 LoRA, v2 — class-imbalance-aware loss + heavier rare oversampling.

Goal: lift recall on Bond_Crack (cls 5), Bond_Open (cls 6), Receptor_Lightning
(cls 7) which sit at 41-53% image-level recall in v1's iter_30000 baseline TTA
eval. Plain CE was letting La_Exposure (~65% of all positive instances) dominate
the gradient.

Changes vs `sam3_lora_windblade_8class.py` (v1) — everything else is unchanged:

  1. loss_decode: single `CrossEntropyLoss` (loss_weight=1.0)
     → list of [class-weighted CE, Dice loss].
       - CWCE: class_weight downweights BG (0.5) and upweights rare defects
         (Bond_Open=4, Bond_Crack/La_Crack/La_Open=3, Receptor=2, La_Damage=1.5).
       - Dice (use_sigmoid=False, multiclass softmax form): loss_weight=1.0,
         ignore_index=255. Pulls gradient toward whole-region overlap, which is
         what mIoU / recall reward. Equally weighted to CE so neither dominates.

  2. train_dataset_rare RepeatDataset times: 5 → 10.
     Combined with RareClassCrop's 0.8 prob, the rare classes now see roughly
     2× more crop exposure per epoch.

  3. work_dir: runs/sam3_lora_v1 → runs/sam3_lora_v2 (so v1 is preserved).

  4. (Optional, easy to flip) FastValHookV2 saves best_recall_*.pth — that hook
     uses macro recall; if you want it driven by micro recall instead, edit
     fast_val_hook_v2.py (but per your "always-new-files" rule, do it in a
     v2 fork of that hook).

Recommended launch: fine-tune from v1's iter_30000.pth rather than train from
scratch. The model is already converged on the v1 loss; v2 just rebalances.
~5-10k extra iters should re-stabilize:

    cd /home/work/workspace/jongwon/dmdd/mmsegmentation
    export PYTHONPATH=/home/work/workspace/azka5/dmdd_pipeline/seg:$PYTHONPATH
    python tools/train.py \\
        /home/work/workspace/azka5/dmdd_pipeline/seg/configs/sam3_lora_windblade_8class_v2.py \\
        --resume /home/work/workspace/azka5/dmdd_pipeline/runs/sam3_lora_v1/iter_30000.pth
"""

_base_ = [
    '/home/work/workspace/jongwon/dmdd/mmsegmentation/configs/_base_/models/segformer_mit-b0.py',
    '/home/work/workspace/jongwon/dmdd/mmsegmentation/configs/_base_/datasets/windblade_7class_with_blade_mask.py',
    '/home/work/workspace/jongwon/dmdd/mmsegmentation/configs/_base_/default_runtime.py',
    '/home/work/workspace/jongwon/dmdd/mmsegmentation/configs/_base_/schedules/schedule_160k.py',
]

custom_imports = dict(
    imports=['_torch_compat', 'sam3_backbone', 'fast_val_hook_v2'],
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

# Per-class weights (9 entries: BG + 8 defect classes).
# Higher = stronger gradient = recall improvement at the cost of precision.
# Tune downward if rare-class FPs explode.
CLASS_WEIGHTS = [
    0.5,  # 0 Background          — downweighted (always dominant in pixel count)
    1.0,  # 1 La_Exposure         — already at ~86% recall, leave neutral
    1.5,  # 2 La_Damage           — at ~85% recall, mild bump
    2.5,  # 3 La_Crack            — at ~69% recall, push up
    3.0,  # 4 La_Open              — no test GT, train present; boost just in case
    3.0,  # 5 Bond_Crack          — at ~58% recall
    4.0,  # 6 Bond_Open           — at ~42% recall, the floor
    2.0,  # 7 Receptor_Lightning  — at ~55% recall
    1.5,  # 8 Receptor_Damage     — at ~79% recall
]

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
                use_sigmoid=False,        # multiclass softmax form
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

# ---- Datasets ----------------------------------------------------------------
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

# v1 used times=5; v2 doubles to times=10 for more rare-class exposure.
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

work_dir = '/home/work/workspace/azka5/dmdd_pipeline/runs/sam3_lora_v2'

import os
os.environ['TZ'] = 'Asia/Seoul'
import time
time.tzset()
