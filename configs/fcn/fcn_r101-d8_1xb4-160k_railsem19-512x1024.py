_base_ = ['./fcn_r101-d8_4xb2-80k_cityscapes-512x1024.py']
dataset_type = 'RailSem19Dataset'
data_root = 'data/RailSem19/'
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(1920, 1080),
        ratio_range=(0.5, 1.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1920, 1080), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='jpgs/rs19_val', seg_map_path='uint8/rs19_val'),
        ann_file='rs19_splits4000/train.txt',
        pipeline=train_pipeline))
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='jpgs/rs19_val', seg_map_path='uint8/rs19_val'),
        ann_file='rs19_splits4000/val.txt',
        pipeline=test_pipeline))
test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='jpgs/rs19_val', seg_map_path='uint8/rs19_val'),
        ann_file='rs19_splits4000/test.txt',
        pipeline=test_pipeline))

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/fcn/fcn_r101-d8_512x1024_80k_cityscapes/fcn_r101-d8_512x1024_80k_cityscapes_20200606_113038-3fb937eb.pth'

# Batch size reduced: batch_size=4, 160k iterations, lr=0.005 (halved for smaller batch)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=8000)
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=160000,
        by_epoch=False)
]
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000))
