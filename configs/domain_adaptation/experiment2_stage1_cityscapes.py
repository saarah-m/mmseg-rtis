_base_ = [
    "../_base_/models/segformer_mit-b0.py",
    "../_base_/datasets/cityscapes_1024x1024.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_160k.py",
]

# Experiment 2 - Stage 1: Train on Cityscapes
# This is the first stage where we pre-train the model on Cityscapes dataset.
# The resulting checkpoint will be used in experiment2_cityscapes_to_railsem19.py

crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)

# Load pretrained weights from ImageNet
checkpoint = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth"

# Enable TensorBoard visualization
vis_backends = [dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")]
visualizer = dict(
    type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)

# Experiment metadata
experiment_name = "experiment2_stage1_cityscapes"
dataset = "cityscapes"

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type="Pretrained", checkpoint=checkpoint),
    ),
    decode_head=dict(
        num_classes=19,  # Cityscapes has 19 classes
        ignore_index=255,
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=1.0
        )
    ),
    test_cfg=dict(mode="slide", crop_size=crop_size, stride=(768, 768)),
)

# Optimizer configuration
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "pos_block": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
            "head": dict(lr_mult=10.0),
        }
    ),
)

# Learning rate schedule - extended to allow training until convergence
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=320000,  # Extended from 160k to 320k to allow convergence
        by_epoch=False,
    ),
]

# Training configuration - extended with early stopping
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=320000,  # Extended to allow full convergence
    val_interval=16000  # Validate every 16k iterations
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Training dataloaders
train_dataloader = dict(batch_size=1, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

# Early Stopping - automatically stops when converged
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='mIoU',
        patience=3,  # Stop if no improvement for 3 validation checks (48k iterations)
        rule='greater',  # We want mIoU to increase
        min_delta=0.2,  # Minimum improvement threshold (0.2% mIoU)
    )
]

# Checkpoint saving - save best model
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=16000,
        max_keep_ckpts=5,  # Keep last 5 checkpoints
        save_best='mIoU',  # Save best model based on mIoU
        rule='greater'
    )
)

