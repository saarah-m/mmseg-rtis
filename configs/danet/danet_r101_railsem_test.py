custom_imports = dict(
    imports=['mmseg.datasets.railsem19'],
    allow_failed_imports=False
)

_base_ = [
    '../_base_/models/danet_r50-d8.py', 
    '../_base_/datasets/railsem19_1024x1024.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/single_schedule.py'
]

crop_size = (512, 1024)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size=crop_size, 
    pad_val=0,
    seg_pad_val=255,
)

model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None, 
    backbone=dict(depth=101)
)


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
    num_workers=2,
    persistent_workers=True,
    dataset=dict(pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(pipeline=test_pipeline)
)

test_dataloader = val_dataloader