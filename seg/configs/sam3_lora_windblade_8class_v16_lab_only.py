"""SAM 3.1 LoRA, v16 — v8 recipe trained ONLY on overseas lab-labeler images.

Lab labelers (the `lab` group in overseas_labels_combined.csv) annotated 6
task types — 3_La_Crack, 3_Bond_Crack, 4_La_Crack, 4_Bond_Crack, 5_La_Crack,
5_Bond_Crack — which collapse to 2 semantic classes: La_Crack (3) and
Bond_Crack (5). So v16 effectively learns Crack-vs-background from the
lab labelers' style only.

Train: 2,993 lab images (from overseas turbine-split train fold) — all
from train_rare (lab images never appeared in train_full).
  defect tiles: 8,131 (4,552 La_Crack + 3,579 Bond_Crack)
  bg tiles:     7,844
  total:       15,975 tiles per epoch

Val   (FastValHookV2):  lab_only_val.json  (345 imgs)
Test  (after training): lab_only_test.json (303 imgs)

Recipe is byte-for-byte v8/v12/v13/v15: CWCE+Dice, LoRA r=8, v1-warmup,
30k iters DDP-2 batch=4. The 9-channel head is unchanged; the model can
still predict other classes — they just won't appear in lab GT during
training, so the loss will only ever push 3 and 5.
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
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)

CLASS_WEIGHTS = [0.5, 1.0, 1.5, 2.5, 3.0, 3.0, 4.0, 2.0, 1.5]

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

azka_data = '/home/work/workspace/azka5/dmdd_pipeline/data'
overseas_blade_dir = f'{azka_data}/overseas/blade_masks_640'

TILE_DEFECT_JSON = f'{azka_data}/train_tiles_defect_lab_only.json'
TILE_BG_JSON     = f'{azka_data}/train_tiles_bg_lab_only.json'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='FixedTileCrop', default_crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

defect_dataset = dict(type='WindBladeTiledDataset', ann_file=TILE_DEFECT_JSON, pipeline=train_pipeline)
bg_dataset     = dict(type='WindBladeTiledDataset', ann_file=TILE_BG_JSON,     pipeline=train_pipeline)

train_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[defect_dataset, bg_dataset],
    ),
)

# Default test = lab test (watcher overrides via --test-json at inference).
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(ann_file=f'{azka_data}/lab_only_test.json'),
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
            name='v16_lab_only_cracks_iter0-30k',
            group='sam3_lora_v16_lab_only',
            tags=['sam3.1', 'lora', 'cwce', 'dice', 'tiles', 'overseas', 'lab-only',
                  'cracks-only', 'v1-warmup', 'turbine-split-overseas'],
            notes='v8 recipe trained on overseas lab-labeler-only images (2993 train/345 val/303 test). Lab labelers annotate Cracks only (La_Crack + Bond_Crack). Eval on lab-test only.',
        ),
    ),
]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

custom_hooks = [
    dict(type='FastValHookV2',
         data_json=f'{azka_data}/lab_only_val.json',
         val_interval=10000,
         num_classes=9,
         initial_val=False,
         blade_filter_mask_dir=overseas_blade_dir,
         blade_filter_classes=(1, 2),
         num_workers=8,
         prefetch_factor=2,
         use_bf16=True,
         persistent_workers=True,
         slide_batch_size=8),
]

work_dir = '/home/work/workspace/azka5/dmdd_pipeline/runs/sam3_lora_v16_lab_only'

load_from = '/home/work/workspace/azka5/dmdd_pipeline/runs/sam3_lora_v1/iter_30000.pth'

import os
os.environ['TZ'] = 'Asia/Seoul'
import time
time.tzset()
