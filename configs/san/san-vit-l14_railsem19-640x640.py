_base_ = [
    '../_base_/models/san_vit-b16.py', '../_base_/datasets/railsem19.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

# Use ViT-L14 pretrained weights from COCO-Stuff164K
pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/san/clip_vit-large-patch14-336_3rdparty-0b5df9cb.pth'

# Adapt crop size for RailSem19 (640x640 for consistency with original)
crop_size = (640, 640)

# Update data preprocessing for RailSem19
data_preprocessor = dict(
    mean=[122.7709, 116.7460, 104.0937],
    std=[68.5005, 66.6322, 70.3232],
    size_divisor=640,
    test_cfg=dict(size_divisor=32))

# Update training pipeline for 640x640 crop size
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomChoiceResize',
        scales=[int(640 * x * 0.1) for x in range(5, 16)],
        resize_type='ResizeShortestEdge',
        max_size=2560),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeShortestEdge', scale=crop_size, max_size=2560),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# Update dataloaders
train_dataloader = dict(batch_size=1, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=1, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# Model configuration for ViT-L14
model = dict(
    type="MultimodalEncoderDecoder",
    pretrained=pretrained,
    encoder_resolution=0.7,
    image_encoder=dict(
        type="VisionTransformer",
        img_size=(336, 336),
        patch_size=14,
        patch_pad=0,
        embed_dims=1024,
        num_layers=18,
        num_heads=16,
        out_indices=(5, 11, 17),
        frozen_exclude=[
            "pos_embed",
            "cls_token",
        ],  # Make position embedding and cls token trainable
    ),
    text_encoder=dict(
        type="CLIPTextEncoder",
        dataset_name="railsem19",  # Update for RailSem19
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        output_dims=768,
        # Note: CLIPTextEncoder doesn't support frozen_exclude, all parameters are frozen by default
    ),
    decode_head=dict(
        type="SideAdapterCLIPHead",
        num_classes=19,  # RailSem19 has 19 classes
        san_cfg=dict(clip_channels=1024, cfg_decoder=dict(num_heads=16)),
        maskgen_cfg=dict(
            num_layers=6,
            embed_dims=1024,
            num_heads=16,
            out_dims=768,
            frozen_exclude=["all"],  # Make all parameters in maskgen trainable
        ),
        loss_decode=[
            dict(type="CrossEntropyLoss", loss_name="loss_cls_ce"),
            dict(type="CrossEntropyLoss", loss_name="loss_mask_ce"),
        ],
    ),
)

# Training schedule for transfer learning (shorter schedule)
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=40000,  # Reduced for transfer learning
    val_interval=500,
    val_begin=35000)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=5000,  # More frequent checkpoints
        save_best='mIoU'))

# Optimizer configuration for transfer learning
# Simulate batch size of 2 using gradient accumulation
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00005, betas=(0.9, 0.999), weight_decay=0.0001),  # Lower learning rate for transfer learning
    paramwise_cfg=dict(
        custom_keys={
            'img_encoder': dict(lr_mult=0.1, decay_mult=1.0),
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    loss_scale='dynamic',
    clip_grad=dict(max_norm=0.01, norm_type=2),
    accumulative_counts=2  # Simulate batch size of 2
)

# Learning rate scheduler for transfer learning
param_scheduler = [
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=0,
        end=40000,  # Match max_iters
        by_epoch=False,
    )
]

# Enable TensorBoard visualization
vis_backends = [dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")]
visualizer = dict(
    type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)
