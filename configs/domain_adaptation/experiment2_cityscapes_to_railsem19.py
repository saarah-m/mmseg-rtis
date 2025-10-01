_base_ = [
    "../_base_/models/segformer_mit-b0.py",
    "../_base_/datasets/railsem19_1024x1024.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_160k.py",
]

# Experiment: CityScapes → RailSem19
# This configuration trains a model on CityScapes and then fine-tunes on RailSem19

crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)

# Load pretrained weights from CityScapes
checkpoint = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth"

# Enable TensorBoard visualization
vis_backends = [dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")]
visualizer = dict(
    type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)

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

# Learning rate schedule for domain adaptation
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
train_dataloader = dict(batch_size=1, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

# Experiment metadata
experiment_name = "cityscapes_to_railsem19"
source_dataset = "cityscapes"
target_dataset = "railsem19"
