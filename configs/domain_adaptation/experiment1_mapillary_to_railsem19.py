_base_ = [
    "../_base_/models/segformer_mit-b0.py",
    "../_base_/datasets/railsem19_1024x1024.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_160k.py",
]

# Experiment 1 - Stage 2: Mapillary → RailSem19
# This configuration fine-tunes a model pre-trained on Mapillary to RailSem19.
# 
# PREREQUISITE: First run experiment1_stage1_mapillary.py to train on Mapillary.
# Then update the checkpoint path below to point to the best checkpoint from stage 1.
# Example: "work_dirs/experiment1_stage1_mapillary/best_mIoU_iter_XXXXX.pth"

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

# Load pretrained weights from Mapillary (Stage 1)
# Use --cfg-options load_from=<path> when running to specify the checkpoint
# Example: --cfg-options load_from=work_dirs/domain_adaptation/experiment1/stage1_mapillary/best_mIoU_iter_*.pth
load_from = None  # Will be set via command line

# Enable TensorBoard visualization
vis_backends = [dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")]
visualizer = dict(
    type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)

# Experiment metadata
experiment_name = "experiment1_stage2_railsem19"
source_dataset = "mapillary"
target_dataset = "railsem19"

# SyncBN for decode head
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=19,  # RailSem19 has 19 classes
        norm_cfg=norm_cfg,
        loss_decode=dict(type="CrossEntropyLoss"),
    ),
    test_cfg=dict(mode="slide", crop_size=(1024, 1024), stride=(768, 768)),
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
            [dict(type='LoadAnnotations')],
            [dict(type='PackSegInputs')]
        ])
]

# Optimizer configuration for domain adaptation with AMP
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
