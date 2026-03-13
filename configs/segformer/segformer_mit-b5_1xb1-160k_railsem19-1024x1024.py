_base_ = ['./segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py']
dataset_type = 'RailSem19Dataset'
data_root = 'data/RailSem19/'
crop_size = (1080, 1920)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(with_cp=True),
    decode_head=dict(num_classes=19),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(1080, 1920)))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomChoiceResize',
        scales=[(int(1080 * 0.5), int(1920 * 0.5)), (1080, 1920), (int(1080 * 2), int(1920 * 2))],
        resize_type='Resize',
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1080, 1920), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/images', seg_map_path='train/annotations'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/images', seg_map_path='val/annotations'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test/images', seg_map_path='test/annotations'),
        pipeline=test_pipeline))

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=8000)
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-6,
        power=2.0,
        begin=0,
        end=160000,
        by_epoch=False)
]
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000))
