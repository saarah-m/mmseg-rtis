# RailSem19 train with truck oversample: same as railsem19 but train = full + train_truck_only
# so images that contain truck (label 14) are seen ~2x per epoch.
# Run once: python tools/misc/build_railsem19_truck_oversample.py
_base_ = './railsem19.py'

dataset_type = _base_.dataset_type
data_root = _base_.data_root
METAINFO = _base_.METAINFO
train_pipeline = _base_.train_pipeline

_train_full = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path='train/images', seg_map_path='train/annotations'),
    img_suffix='.jpg',
    seg_map_suffix='.png',
    pipeline=train_pipeline,
    metainfo=METAINFO,
)
_train_truck = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(
        img_path='train_truck_only/images',
        seg_map_path='train_truck_only/annotations',
    ),
    img_suffix='.jpg',
    seg_map_suffix='.png',
    pipeline=train_pipeline,
    metainfo=METAINFO,
)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(type='ConcatDataset', datasets=[_train_full, _train_truck]),
)
