_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/rtis_rail.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(
    size=crop_size,
    pad_val=0,
    seg_pad_val=255  
)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(dilations=(1, 1, 2, 4), strides=(1, 2, 2, 2)))

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
)

test_dataloader = val_dataloader


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(2048, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),  
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=2,
    dataset=dict(pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=1,
    dataset=dict(pipeline=test_pipeline)
)

test_dataloader = val_dataloader