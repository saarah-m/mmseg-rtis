

_base_ = [
    '../_base_/datasets/rtis_rail.py', '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]



crop_size = (512, 1024)

NUM_CLASSES = 19
model = dict(
    _delete_=True,  # remove inherited base model
    type='EncoderDecoder',
    
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnet50_v1c')
    ),
    
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=NUM_CLASSES,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_classes=NUM_CLASSES,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    ),
    
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    
 data_preprocessor=dict(
    type='SegDataPreProcessor',
    size=None,
    size_divisor=32,  # ⚠ use size_divisor here, but **no fixed crop** in pipeline
    pad_val=0,
    seg_pad_val=255,
    bgr_to_rgb=True,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375]
)
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
    )
)