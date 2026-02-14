# OCRNet + HRNet-W18 on RailSem19 (base for W18-Small variant)
_base_ = [
    '../_base_/models/ocrnet_hr18.py', '../_base_/datasets/railsem19.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)

# TensorBoard
vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
