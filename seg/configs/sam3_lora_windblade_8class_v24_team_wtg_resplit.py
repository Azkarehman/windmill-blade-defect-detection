"""SAM 3.1 LoRA, v24 — team train_no_leak re-split 80/10/10 by WTG.

Differs from v12 in WTG-disjoint splits (v12 used the team's standard val_diuid_7
and test_diuid_10 which are NOT WTG-disjoint w.r.t. train). v24 ensures every
train WTG is held out from val/test.

Recipe: v12 byte-for-byte (CWCE+Dice, LoRA, --n-bg=3, no BG upsampling, 30k iters).

Train: 7,737 imgs (200 WTGs)
Val:    961 imgs (25 WTGs)
Test:   866 imgs (26 WTGs)
"""
_base_ = [
    '/home/work/workspace/jongwon/dmdd/mmsegmentation/configs/_base_/models/segformer_mit-b0.py',
    '/home/work/workspace/jongwon/dmdd/mmsegmentation/configs/_base_/datasets/windblade_7class_with_blade_mask.py',
    '/home/work/workspace/jongwon/dmdd/mmsegmentation/configs/_base_/default_runtime.py',
    '/home/work/workspace/jongwon/dmdd/mmsegmentation/configs/_base_/schedules/schedule_160k.py',
]

custom_imports = dict(
    imports=['_torch_compat', 'sam3_backbone', 'fast_val_hook_v2',
             'windblade_tiled', 'fixed_tile_crop'],
    allow_failed_imports=False,
)

crop_size = (1008, 1008)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True, pad_val=0, seg_pad_val=255, size=crop_size,
)

CLASS_WEIGHTS = [0.5, 1.0, 1.5, 2.5, 3.0, 3.0, 4.0, 2.0, 1.5]

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(_delete_=True, type='Sam3Backbone',
                  lora_r=8, lora_alpha=16, lora_dropout=0.1,
                  freeze_trunk_other_than_lora=True, train_neck=True),
    decode_head=dict(
        in_channels=[256, 256, 256, 256], in_index=[0, 1, 2, 3],
        num_classes=9,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
                 class_weight=CLASS_WEIGHTS, loss_name='loss_cwce'),
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=1.0,
                 ignore_index=255, eps=1e-5, loss_name='loss_dice'),
        ],
    ),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(504, 504)),
)

optim_wrapper = dict(
    _delete_=True, type='AmpOptimWrapper', dtype='bfloat16',
    optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'pos_block': dict(decay_mult=0.),
        'norm': dict(decay_mult=0.),
        'decode_head': dict(lr_mult=10.),
    }),
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

azka_data = '/home/work/workspace/azka5/dmdd_pipeline/data'
blade_mask_dir = '/home/work/workspace/jongwon/dmdd/data/blade_masks_640'

TILE_DEFECT_JSON = f'{azka_data}/train_tiles_defect_v24_wtg_resplit.json'
TILE_BG_JSON     = f'{azka_data}/train_tiles_bg_v24_wtg_resplit.json'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='FixedTileCrop', default_crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

defect_ds = dict(type='WindBladeTiledDataset', ann_file=TILE_DEFECT_JSON, pipeline=train_pipeline)
bg_ds     = dict(type='WindBladeTiledDataset', ann_file=TILE_BG_JSON,     pipeline=train_pipeline)

train_dataloader = dict(
    _delete_=True, batch_size=4, num_workers=8, persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(type='ConcatDataset', datasets=[defect_ds, bg_ds]),
)

test_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=True,
    dataset=dict(ann_file=f'{azka_data}/team_wtg_resplit_test.json'),
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
    dict(type='WandbVisBackend', init_kwargs=dict(
        project='dmdd', entity='usmanfamily',
        name='v24_team_wtg_resplit_iter0-30k',
        group='sam3_lora_v24_team_wtg_resplit',
        tags=['sam3.1','lora','cwce','dice','tiles','team-no-leak','wtg-disjoint-resplit','v12-recipe'],
        notes='v12 recipe with team train_no_leak re-split 80/10/10 WTG-disjoint. Clean baseline avoiding diuid-hash leakage in team default val/test.',
    )),
]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

custom_hooks = [
    dict(type='FastValHookV2',
         data_json=f'{azka_data}/team_wtg_resplit_val.json',
         val_interval=10000, num_classes=9, initial_val=False,
         blade_filter_mask_dir=blade_mask_dir, blade_filter_classes=(1, 2),
         num_workers=8, prefetch_factor=2, use_bf16=True,
         persistent_workers=True, slide_batch_size=8),
]

work_dir = '/home/work/workspace/azka5/dmdd_pipeline/runs/sam3_lora_v24_team_wtg_resplit'
load_from = '/home/work/workspace/azka5/dmdd_pipeline/runs/sam3_lora_v1/iter_30000.pth'

import os
os.environ['TZ'] = 'Asia/Seoul'
import time
time.tzset()
