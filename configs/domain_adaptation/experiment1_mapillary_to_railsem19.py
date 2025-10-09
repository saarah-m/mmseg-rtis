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
data_preprocessor = dict(size=crop_size)

# Load pretrained weights from Mapillary (Stage 1)
# TODO: Update this path after running experiment1_stage1_mapillary.py
checkpoint = "work_dirs/experiment1_stage1_mapillary/best_mIoU_iter_XXXXX.pth"
# If you haven't run stage 1 yet, temporarily use ImageNet weights:
# checkpoint = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth"

# Enable TensorBoard visualization
vis_backends = [dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")]
visualizer = dict(
    type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)

# Experiment metadata
experiment_name = "experiment1_stage2_railsem19"
source_dataset = "mapillary"
target_dataset = "railsem19"

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type="Pretrained", checkpoint=checkpoint),
    ),
    decode_head=dict(
        num_classes=19,  # RailSem19 has 19 classes
        loss_decode=dict(type="CrossEntropyLoss"),
    ),
    test_cfg=dict(mode="slide", crop_size=(1024, 1024), stride=(768, 768)),
)

# Optimizer configuration for domain adaptation
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "pos_block": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
            "head": dict(lr_mult=10.0),  # Higher learning rate for head
        }
    ),
)

# Learning rate schedule for domain adaptation - extended to allow convergence
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
        rule='greater',
        min_delta=0.2,  # Minimum improvement threshold (0.2% mIoU)
    )
]

# Checkpoint saving - save best model
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=16000,
        max_keep_ckpts=5,
        save_best='mIoU',
        rule='greater'
    )
)
