#!/bin/bash
# train_all_stages.sh - Using CUDA device 6

export CUDA_VISIBLE_DEVICES=6

# Stage 1: Mapillary from scratch
echo "Starting Stage 1: Mapillary from scratch on GPU 6..."
python tools/train.py configs/mask2former/mask2former_swin-l-in22k-224x224-pre_8xb2-160k_mapillary-1024x1024.py \
    --work-dir work_dirs/mask2former_swin-l_mapillary_from_scratch

# Stage 2: Mapillary → Cityscapes
STAGE1_CKPT=$(ls -t work_dirs/mask2former_swin-l_mapillary_from_scratch/best_mIoU_iter_*.pth | head -1)
echo "Starting Stage 2: Mapillary → Cityscapes (loading from $STAGE1_CKPT)..."
python tools/train.py configs/mask2former/mask2former_swin-l-in22k-224x224-pre_8xb2-160k_mapillary_to_cityscapes-512x1024.py \
    --work-dir work_dirs/mask2former_swin-l_mapillary_to_cityscapes \
    --cfg-options load_from=$STAGE1_CKPT

# Stage 3: Cityscapes → RailSem19
STAGE2_CKPT=$(ls -t work_dirs/mask2former_swin-l_mapillary_to_cityscapes/best_mIoU_iter_*.pth | head -1)
echo "Starting Stage 3: Cityscapes → RailSem19 (loading from $STAGE2_CKPT)..."
python tools/train.py configs/mask2former/mask2former_swin-l-in22k-224x224-pre_8xb2-160k_cityscapes_to_railsem19-1024x1024.py \
    --work-dir work_dirs/mask2former_swin-l_cityscapes_to_railsem19 \
    --cfg-options load_from=$STAGE2_CKPT

echo "All stages completed!"