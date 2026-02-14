# FCN HRNet-W18 on RailSem19 (base for HRNet-W18-Small variant)
_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/railsem19.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)

# Mild class weights for rare classes (cap 3 to avoid hurting common classes)
# Boosts rare: traffic-light, traffic-sign, human, truck, rail-embedded
class_weight = [
    0.9, 1.0, 0.5, 2.0, 1.0, 0.95, 3.0, 3.0, 0.4, 0.7,
    0.5, 3.0, 0.7, 2.0, 3.0, 0.5, 1.5, 1.0, 3.0,
]
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=class_weight,
        )))

# TensorBoard logging (view with: tensorboard --logdir work_dirs/<exp>/vis_data)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)
