"""Drop-in SAM-3.1 replacement for SegFormer-B4 in the team's pipeline.

Same data pipeline, same transforms (BladeFilteredRandomCrop, RareClassCrop,
PhotoMetricDistortion, ConcatDataset(full + RepeatDataset×5(rare))), same
optimizer/schedule shape (AdamW + LinearLR warmup + PolyLR), 9-class output.

Differences from `segformer_mit-b4_windblade_8class_blade_filter.py`:
  - Backbone: mit_b4 → Sam3Backbone (LoRA r=8 on attn.qkv/proj, neck trainable,
    trunk otherwise frozen). ~9.4M of 455M params trainable.
  - crop_size: 1024 → 1008  (SAM 3.1's RoPE is fixed at 1008²).
  - Decode head in_channels: [64, 128, 320, 512] → [256, 256, 256, 256].
  - max_iters: 300k → 30k. Foundation-model fine-tuning needs far fewer iters.
  - batch_size: 4 → 8 (room thanks to mostly-frozen encoder + LoRA).
  - data_root + work_dir set for the KT Cloud A100/H200 layout.
  - FastValHook → FastValHookV2 (DataLoader workers + bf16 + batched slide).

Usage:
    cd /home/work/workspace/jongwon/dmdd/mmsegmentation
    export PYTHONPATH=/home/work/workspace/azka5/dmdd_pipeline/seg:$PYTHONPATH
    python tools/train.py \\
        /home/work/workspace/azka5/dmdd_pipeline/seg/configs/sam3_lora_windblade_8class.py \\
        --resume /home/work/workspace/azka5/dmdd_pipeline/runs/sam3_lora_v1/iter_5000.pth
"""

# Absolute paths to the team's mmsegmentation _base_ configs.
# (mmengine extracts `_base_` via AST — variables/f-strings here aren't
# evaluated, so paths must be string literals.)
_base_ = [
    '/home/work/workspace/jongwon/dmdd/mmsegmentation/configs/_base_/models/segformer_mit-b0.py',
    '/home/work/workspace/jongwon/dmdd/mmsegmentation/configs/_base_/datasets/windblade_7class_with_blade_mask.py',
    '/home/work/workspace/jongwon/dmdd/mmsegmentation/configs/_base_/default_runtime.py',
    '/home/work/workspace/jongwon/dmdd/mmsegmentation/configs/_base_/schedules/schedule_160k.py',
]

# Make our Sam3Backbone and FastValHookV2 visible to mmseg's registries.
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
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    ),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(504, 504)),
)

# ---- Optimizer / schedule ----------------------------------------------------
# Same shape as team config, but the head:LR-multiplier matters less now that
# the encoder is mostly frozen and only LoRA + neck train.
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',   # bf16 training; H200 bf16 tensor cores → ~1.5×
    dtype='bfloat16',
    optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'decode_head': dict(lr_mult=10.),
            # LoRA adapter params get default LR (no boost — they're already
            # initialized small and don't need the 10× head LR).
        },
    ),
)

# 30k iters total, with 500-iter warmup (proportionally shorter than team's 1500
# warmup for 300k iters).
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

train_dataset_rare = dict(
    type='RepeatDataset',
    times=5,
    dataset=dict(
        type='WindBlade7ClassDataset',
        ann_file=f'{data_root}/train_rare.json',
        pipeline=train_pipeline_rare,
    ),
)

train_dataloader = dict(
    _delete_=True,
    # 4 per GPU × 2 GPUs (DDP) = effective batch 8 — preserves single-GPU
    # training dynamics. If running single-GPU, set this back to 8.
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

# Faster validation: DataLoader workers do parallel image / GT / blade-mask
# I/O, autocast(bf16) halves per-window cost, batched slide cuts kernel-
# launch overhead. Best-checkpoint selection (best_recall_iter_*.pth and
# best_mIoU_iter_*.pth) is preserved.
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

work_dir = '/home/work/workspace/azka5/dmdd_pipeline/runs/sam3_lora_v1'

import os
os.environ['TZ'] = 'Asia/Seoul'
import time
time.tzset()
