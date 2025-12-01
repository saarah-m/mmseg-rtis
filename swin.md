<!-- STAGE ONE -->

export CUDA_VISIBLE_DEVICES=5,6,7
bash tools/dist_train.sh configs/mask2former/mask2former_swin-l-in22k-224x224-pre_8xb2-160k_mapillary-1024x1024.py 3 \
    --work-dir work_dirs/mask2former_swin-l_mapillary_from_scratch


