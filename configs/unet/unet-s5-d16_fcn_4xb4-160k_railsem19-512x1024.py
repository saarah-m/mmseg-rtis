_base_ = ['./unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py']
dataset_type = 'RailSem19Dataset'
data_root = 'data/RailSem19/'
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='jpgs/rs19_val', seg_map_path='uint8/rs19_val'),
        ann_file='rs19_splits4000/train.txt'))
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='jpgs/rs19_val', seg_map_path='uint8/rs19_val'),
        ann_file='rs19_splits4000/val.txt'))
test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='jpgs/rs19_val', seg_map_path='uint8/rs19_val'),
        ann_file='rs19_splits4000/test.txt'))

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/unet/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth'

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    accumulative_counts=8,
    clip_grad=None)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=1280000, val_interval=128000)
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=1280000,
        by_epoch=False)
]
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=128000))

model = dict(backbone=dict(with_cp=True))
