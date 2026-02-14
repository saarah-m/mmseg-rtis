# OCRNet + HRNetV2p-W18-Small on RailSem19, transfer from Cityscapes OCRNet.
# Test with multi-scale + flip (like NVIDIA multi-scale): python tools/test.py CONFIG CHECKPOINT --tta
_base_ = './ocrnet_hr18_4xb2-40k_railsem19-512x1024.py'
load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18s_4xb2-40k_cityscapes-512x1024/ocrnet_hr18s_4xb2-40k_cityscapes-512x1024_20230227_145026-6c052a14.pth'
model = dict(
    pretrained=None,
    backbone=dict(
        extra=dict(
            stage1=dict(num_blocks=(2, )),
            stage2=dict(num_blocks=(2, 2)),
            stage3=dict(num_modules=3, num_blocks=(2, 2, 2)),
            stage4=dict(num_modules=2, num_blocks=(2, 2, 2, 2)))))
