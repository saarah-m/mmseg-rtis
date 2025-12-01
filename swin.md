export CUDA_VISIBLE_DEVICES=5,6,7

<!-- # Stage 1: Mapillary from scratch -->
bash tools/dist_train.sh configs/mask2former/mask2former_swin-l-in22k-224x224-pre_8xb2-160k_mapillary-1024x1024.py 3 \
    --work-dir work_dirs/mask2former_swin-l_mapillary_from_scratch

<!-- # Stage 2: Mapillary → Cityscapes -->
bash tools/dist_train.sh configs/mask2former/mask2former_swin-l-in22k-224x224-pre_8xb2-160k_mapillary_to_cityscapes-512x1024.py 3 \
    --work-dir work_dirs/mask2former_swin-l_mapillary_to_cityscapes \
    --cfg-options load_from=<path_to_stage1_checkpoint>

<!-- # Stage 3: Cityscapes → RailSem19 -->
bash tools/dist_train.sh configs/mask2former/mask2former_swin-l-in22k-224x224-pre_8xb2-160k_cityscapes_to_railsem19-1024x1024.py 3 \
    --work-dir work_dirs/mask2former_swin-l_cityscapes_to_railsem19 \
    --cfg-options load_from=<path_to_stage2_checkpoint>

