# FCN HRNetV2p-W18-Small on RailSem19, transfer learning from Cityscapes pretrained
_base_ = './fcn_hr18_4xb2-40k_railsem19-512x1024.py'
# Cityscapes pretrained checkpoint (same architecture, 19 classes)
load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr18s_512x1024_40k_cityscapes/fcn_hr18s_512x1024_40k_cityscapes_20200601_014216-93db27d0.pth'
# Backbone architecture HRNetV2p-W18-Small; weights from Cityscapes via load_from
model = dict(
    pretrained=None,
    backbone=dict(
        extra=dict(
            stage1=dict(num_blocks=(2, )),
            stage2=dict(num_blocks=(2, 2)),
            stage3=dict(num_modules=3, num_blocks=(2, 2, 2)),
            stage4=dict(num_modules=2, num_blocks=(2, 2, 2, 2)))))
