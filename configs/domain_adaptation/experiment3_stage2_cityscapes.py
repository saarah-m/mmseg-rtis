_base_ = [
    "../_base_/models/segformer_mit-b0.py",
    "../_base_/datasets/cityscapes_1024x1024.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_160k.py",
]

# This configuration is the second stage of Experiment 3
# It fine-tunes a model (pre-trained on Mapillary) on the Cityscapes dataset.

crop_size = (1024, 1024)

# Data preprocessor with ImageNet normalization and bgr_to_rgb
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],  # ImageNet mean (RGB after bgr_to_rgb)
    std=[58.395, 57.12, 57.375],     # ImageNet std (RGB)
    bgr_to_rgb=True,
    size=crop_size,
    pad_val=0,
    seg_pad_val=255
)

# Enable TensorBoard visualization
vis_backends = [dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")]
visualizer = dict(
    type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)

# SyncBN for decode head (backbone retains norm from loaded checkpoint)
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=19,  # Cityscapes has 19 classes
        norm_cfg=norm_cfg,
        ignore_index=255,
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=1.0
        )
    ),
    test_cfg=dict(mode="slide", crop_size=crop_size, stride=(768, 768)),
)

# Optional: TTA for multi-scale + flip
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='Resize', scale_factor=s, keep_ratio=True)
             for s in [0.75, 1.0, 1.25]],
            [dict(type='RandomFlip', prob=0.0), dict(type='RandomFlip', prob=1.0)],
            # Remove LoadAnnotations from TTA - not needed for inference
            [dict(type='PackSegInputs')]
        ])
]

# Optimizer configuration with AMP
optim_wrapper = dict(
    _delete_=True,
    type="AmpOptimWrapper",  # Enable automatic mixed precision
    optimizer=dict(type="AdamW", lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    accumulative_counts=8,  # Effective batch size ~8 to match 6e-5 LR
    paramwise_cfg=dict(
        custom_keys={
            "pos_block": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
            "head": dict(lr_mult=10.0),
        }
    ),
)

# Learning rate schedule
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    ),
]

# Training configuration
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=160000,
    val_interval=8000  # Validate every 8k iterations
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Training dataloaders - base config has proper augmentations
train_dataloader = dict(
    batch_size=1, 
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=1, 
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = val_dataloader

# Checkpoint saving - save best model
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=8000,
        max_keep_ckpts=3,
        save_best='mIoU',
        rule='greater'
    )
)
