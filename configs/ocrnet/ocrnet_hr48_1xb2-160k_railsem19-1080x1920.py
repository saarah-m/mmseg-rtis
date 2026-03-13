_base_ = ['./ocrnet_hr48_4xb2-160k_cityscapes-512x1024.py']
dataset_type = 'RailSem19Dataset'
data_root = 'data/RailSem19/'
crop_size = (1080, 1920)
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
    dict(type='GaussianBlur', sigma_range=(0.15, 1.3), prob=0.5),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=0.9,
                contrast_limit=0.0,
                p=0.5)
        ],
        keymap=dict(img='image', gt_seg_map='mask')),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1080, 1920), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
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

load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr48_512x1024_160k_cityscapes/ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037-dfbf1b0c.pth'

optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

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
