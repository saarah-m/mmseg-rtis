_base_ = ['./fcn_r101-d8_4xb2-80k_cityscapes-512x1024.py']
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

load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/fcn/fcn_r101-d8_512x1024_80k_cityscapes/fcn_r101-d8_512x1024_80k_cityscapes_20200606_113038-3fb937eb.pth'


optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

model = dict(backbone=dict(norm_eval=True))
