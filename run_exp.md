# Experiment 3: Mapillary → Cityscapes → RailSem19

## Stage 1: Pre-train on Mapillary (66 classes)
```bash
python tools/train.py configs/domain_adaptation/experiment3_stage1_mapillary.py \
    --work-dir work_dirs/domain_adaptation/experiment3/stage1_mapillary
```

## Stage 2: Fine-tune on Cityscapes (19 classes)
```bash
python tools/train.py configs/domain_adaptation/experiment3_stage2_cityscapes.py \
    --work-dir work_dirs/domain_adaptation/experiment3/stage2_cityscapes \
    --cfg-options load_from=work_dirs/domain_adaptation/experiment3/stage1_mapillary/iter_160000.pth
```

## Stage 3: Fine-tune on RailSem19 (19 classes)
```bash
python tools/train.py configs/domain_adaptation/experiment3_stage3_railsem19.py \
    --work-dir work_dirs/domain_adaptation/experiment3/stage3_railsem19 \
    --cfg-options load_from=work_dirs/domain_adaptation/experiment3/stage2_cityscapes/iter_160000.pth
```

## Resume Training (if interrupted)
```bash
# Resume Stage 1
python tools/train.py configs/domain_adaptation/experiment3_stage1_mapillary.py \
    --work-dir work_dirs/domain_adaptation/experiment3/stage1_mapillary \
    --resume

# Resume Stage 2
python tools/train.py configs/domain_adaptation/experiment3_stage2_cityscapes.py \
    --work-dir work_dirs/domain_adaptation/experiment3/stage2_cityscapes \
    --resume

# Resume Stage 3
python tools/train.py configs/domain_adaptation/experiment3_stage3_railsem19.py \
    --work-dir work_dirs/domain_adaptation/experiment3/stage3_railsem19 \
    --resume
```
