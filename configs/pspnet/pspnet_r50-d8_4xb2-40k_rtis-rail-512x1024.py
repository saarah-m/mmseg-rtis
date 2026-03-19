_base_ = [
    "../_base_/models/pspnet_r50-d8.py",
    "../_base_/datasets/rtis_rail.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_40k.py",
]

# Modify the model configuration
model = dict(
    data_preprocessor=dict(size=(512, 1024)),
    decode_head=dict(
        num_classes=19,  # RTIS-Rail has 19 classes
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
)

# Modify the training configuration
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

# Modify the optimizer configuration
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005),
    clip_grad=None,
)

# Modify the learning rate scheduler
param_scheduler = [
    dict(type="PolyLR", eta_min=1e-4, power=0.9, begin=0, end=40000, by_epoch=False)
]

# Modify the training schedule
train_cfg = dict(type="IterBasedTrainLoop", max_iters=40000, val_interval=4000)
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=4000),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
