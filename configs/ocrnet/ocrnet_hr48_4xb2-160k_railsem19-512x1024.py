_base_ = ['./ocrnet_hr48_4xb2-160k_cityscapes-512x1024.py']
dataset_type = 'RailSem19Dataset'
data_root = 'data/RailSem19/'
train_dataloader = dict(
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

load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr48_512x1024_160k_cityscapes/ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037-dfbf1b0c.pth'


optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

model = dict(backbone=dict(norm_eval=True))
