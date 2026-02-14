# Same as fcn_hr18s_4xb2-40k_railsem19-512x1024 but oversample images that contain truck.
# Run once before training: python tools/misc/build_railsem19_truck_oversample.py
_base_ = './fcn_hr18s_4xb2-40k_railsem19-512x1024.py'

# Override train_dataloader only: full train + train_truck_only (ConcatDataset)
_train_full = dict(
    type='BaseSegDataset',
    data_root='data/RailSem19/',
    data_prefix=dict(img_path='train/images', seg_map_path='train/annotations'),
    img_suffix='.jpg',
    seg_map_suffix='.png',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='RandomResize', scale=(2048, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
        dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='PackSegInputs'),
    ],
    metainfo=dict(
        classes=('road', 'sidewalk', 'construction', 'tram-track', 'fence',
                 'pole', 'traffic-light', 'traffic-sign', 'vegetation',
                 'terrain', 'sky', 'human', 'rail-track', 'car', 'truck',
                 'trackbed', 'on-rails', 'rail-raised', 'rail-embedded'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [192, 0, 128],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [230, 150, 140], [0, 0, 142], [0, 0, 70],
                 [90, 40, 40], [0, 80, 100], [0, 254, 254], [0, 68, 63]],
    ),
)
_train_truck = dict(
    type='BaseSegDataset',
    data_root='data/RailSem19/',
    data_prefix=dict(
        img_path='train_truck_only/images',
        seg_map_path='train_truck_only/annotations',
    ),
    img_suffix='.jpg',
    seg_map_suffix='.png',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='RandomResize', scale=(2048, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
        dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='PackSegInputs'),
    ],
    metainfo=dict(
        classes=('road', 'sidewalk', 'construction', 'tram-track', 'fence',
                 'pole', 'traffic-light', 'traffic-sign', 'vegetation',
                 'terrain', 'sky', 'human', 'rail-track', 'car', 'truck',
                 'trackbed', 'on-rails', 'rail-raised', 'rail-embedded'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [192, 0, 128],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [230, 150, 140], [0, 0, 142], [0, 0, 70],
                 [90, 40, 40], [0, 80, 100], [0, 254, 254], [0, 68, 63]],
    ),
)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(type='ConcatDatasetIgnoreExtra', datasets=[_train_full, _train_truck]),
)
