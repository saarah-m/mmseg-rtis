# Domain Adaptation Experiments

All experiments now support:
- **Extended training**: Up to 320k iterations (auto-stops when converged)
- **Early stopping**: Stops if no 0.2% mIoU improvement for 48k iterations
- **Best model saving**: Automatically saves checkpoint with highest mIoU
- **TensorBoard logging**: Full visualization support

## Experiment Overview

```
Experiment 1: ImageNet → Mapillary (66 cls) → RailSem19 (19 cls)
              └─ Stage 1 ──────────────────┘   └─ Stage 2 ─────┘

Experiment 2: ImageNet → Cityscapes (19 cls) → RailSem19 (19 cls)
              └─ Stage 1 ────────────────────┘  └─ Stage 2 ─────┘

Experiment 3: ImageNet → Mapillary (66 cls) → Cityscapes (19 cls) → RailSem19 (19 cls)
              └─ Stage 1 ──────────────────┘   └─ Stage 2 ───────┘   └─ Stage 3 ─────┘
```

---

## Experiment 1: Mapillary → RailSem19

### Stage 1: Pre-train on Mapillary (66 classes)
```bash
python tools/train.py configs/domain_adaptation/experiment1_stage1_mapillary.py \
    --work-dir work_dirs/domain_adaptation/experiment1/stage1_mapillary
```

### Stage 2: Fine-tune on RailSem19 (19 classes)
```bash
# First, update the checkpoint path in experiment1_mapillary_to_railsem19.py with the best checkpoint from Stage 1
# Then run:
python tools/train.py configs/domain_adaptation/experiment1_mapillary_to_railsem19.py \
    --work-dir work_dirs/domain_adaptation/experiment1/stage2_railsem19
```

## Experiment 2: Cityscapes → RailSem19

### Stage 1: Pre-train on Cityscapes (19 classes)
```bash
python tools/train.py configs/domain_adaptation/experiment2_stage1_cityscapes.py \
    --work-dir work_dirs/domain_adaptation/experiment2/stage1_cityscapes
```

### Stage 2: Fine-tune on RailSem19 (19 classes)
```bash
# First, update the checkpoint path in experiment2_cityscapes_to_railsem19.py with the best checkpoint from Stage 1
# Then run:
python tools/train.py configs/domain_adaptation/experiment2_cityscapes_to_railsem19.py \
    --work-dir work_dirs/domain_adaptation/experiment2/stage2_railsem19
```

## Experiment 3: Mapillary → Cityscapes → RailSem19

### Stage 1: Pre-train on Mapillary (66 classes)
```bash
python tools/train.py configs/domain_adaptation/experiment3_stage1_mapillary.py \
    --work-dir work_dirs/domain_adaptation/experiment3/stage1_mapillary
```

### Stage 2: Fine-tune on Cityscapes (19 classes)
```bash
# Use the best checkpoint from Stage 1
python tools/train.py configs/domain_adaptation/experiment3_stage2_cityscapes.py \
    --work-dir work_dirs/domain_adaptation/experiment3/stage2_cityscapes \
    --cfg-options load_from=work_dirs/domain_adaptation/experiment3/stage1_mapillary/best_mIoU_iter_*.pth
```

### Stage 3: Fine-tune on RailSem19 (19 classes)
```bash
# Use the best checkpoint from Stage 2
python tools/train.py configs/domain_adaptation/experiment3_stage3_railsem19.py \
    --work-dir work_dirs/domain_adaptation/experiment3/stage3_railsem19 \
    --cfg-options load_from=work_dirs/domain_adaptation/experiment3/stage2_cityscapes/best_mIoU_iter_*.pth
```

---

## Resume/Continue Training from Existing Checkpoints

If you want to continue training from where you left off (e.g., from 160k to convergence):

### Experiment 1 - Stage 1
```bash
python tools/train.py configs/domain_adaptation/experiment1_stage1_mapillary.py \
    --work-dir work_dirs/domain_adaptation/experiment1/stage1_mapillary \
    --cfg-options load_from=work_dirs/domain_adaptation/experiment1/stage1_mapillary/iter_160000.pth
```

### Experiment 1 - Stage 2
```bash
python tools/train.py configs/domain_adaptation/experiment1_mapillary_to_railsem19.py \
    --work-dir work_dirs/domain_adaptation/experiment1/stage2_railsem19 \
    --cfg-options load_from=work_dirs/domain_adaptation/experiment1/stage2_railsem19/iter_160000.pth
```

### Experiment 2 - Stage 1
```bash
python tools/train.py configs/domain_adaptation/experiment2_stage1_cityscapes.py \
    --work-dir work_dirs/domain_adaptation/experiment2/stage1_cityscapes \
    --cfg-options load_from=work_dirs/domain_adaptation/experiment2/stage1_cityscapes/iter_160000.pth
```

### Experiment 2 - Stage 2
```bash
python tools/train.py configs/domain_adaptation/experiment2_cityscapes_to_railsem19.py \
    --work-dir work_dirs/domain_adaptation/experiment2/stage2_railsem19 \
    --cfg-options load_from=work_dirs/domain_adaptation/experiment2/stage2_railsem19/iter_160000.pth
```

### Experiment 3 - Stage 1
```bash
python tools/train.py configs/domain_adaptation/experiment3_stage1_mapillary.py \
    --work-dir work_dirs/domain_adaptation/experiment3/stage1_mapillary \
    --cfg-options load_from=work_dirs/domain_adaptation/experiment3/stage1_mapillary/iter_160000.pth
```

### Experiment 3 - Stage 2
```bash
python tools/train.py configs/domain_adaptation/experiment3_stage2_cityscapes.py \
    --work-dir work_dirs/domain_adaptation/experiment3/stage2_cityscapes \
    --cfg-options load_from=work_dirs/domain_adaptation/experiment3/stage2_cityscapes/iter_160000.pth
```

### Experiment 3 - Stage 3
```bash
python tools/train.py configs/domain_adaptation/experiment3_stage3_railsem19.py \
    --work-dir work_dirs/domain_adaptation/experiment3/stage3_railsem19 \
    --cfg-options load_from=work_dirs/domain_adaptation/experiment3/stage3_railsem19/iter_160000.pth
```

---

## Testing Models

### Experiment 1: Test all stages

**Stage 1 - Test on Mapillary validation set:**
```bash
python tools/test.py \
    configs/domain_adaptation/experiment1_stage1_mapillary.py \
    work_dirs/domain_adaptation/experiment1/stage1_mapillary/best_mIoU_iter_*.pth
```

**Stage 2 - Test on RailSem19 (final model):**
```bash
python tools/test.py \
    configs/domain_adaptation/experiment1_mapillary_to_railsem19.py \
    work_dirs/domain_adaptation/experiment1/stage2_railsem19/best_mIoU_iter_*.pth
```

### Experiment 2: Test all stages

**Stage 1 - Test on Cityscapes validation set:**
```bash
python tools/test.py \
    configs/domain_adaptation/experiment2_stage1_cityscapes.py \
    work_dirs/domain_adaptation/experiment2/stage1_cityscapes/best_mIoU_iter_*.pth
```

**Stage 2 - Test on RailSem19 (final model):**
```bash
python tools/test.py \
    configs/domain_adaptation/experiment2_cityscapes_to_railsem19.py \
    work_dirs/domain_adaptation/experiment2/stage2_railsem19/best_mIoU_iter_*.pth
```

### Experiment 3: Test all stages

**Stage 1 - Test on Mapillary validation set:**
```bash
python tools/test.py \
    configs/domain_adaptation/experiment3_stage1_mapillary.py \
    work_dirs/domain_adaptation/experiment3/stage1_mapillary/best_mIoU_iter_*.pth
```

**Stage 2 - Test on Cityscapes validation set:**
```bash
python tools/test.py \
    configs/domain_adaptation/experiment3_stage2_cityscapes.py \
    work_dirs/domain_adaptation/experiment3/stage2_cityscapes/best_mIoU_iter_*.pth
```

**Stage 3 - Test on RailSem19 (final model):**
```bash
python tools/test.py \
    configs/domain_adaptation/experiment3_stage3_railsem19.py \
    work_dirs/domain_adaptation/experiment3/stage3_railsem19/best_mIoU_iter_*.pth
```

### Cross-Domain Testing (Optional)

Test intermediate models on RailSem19 to measure domain gap:

**Test Mapillary model (Exp 1 Stage 1) on RailSem19:**
```bash
python tools/test.py \
    configs/domain_adaptation/experiment1_mapillary_to_railsem19.py \
    work_dirs/domain_adaptation/experiment1/stage1_mapillary/best_mIoU_iter_*.pth
```

**Test Cityscapes model (Exp 2 Stage 1) on RailSem19:**
```bash
python tools/test.py \
    configs/domain_adaptation/experiment2_cityscapes_to_railsem19.py \
    work_dirs/domain_adaptation/experiment2/stage1_cityscapes/best_mIoU_iter_*.pth
```

**Test Mapillary model (Exp 3 Stage 1) on RailSem19:**
```bash
python tools/test.py \
    configs/domain_adaptation/experiment3_stage3_railsem19.py \
    work_dirs/domain_adaptation/experiment3/stage1_mapillary/best_mIoU_iter_*.pth
```

**Test Cityscapes model (Exp 3 Stage 2) on RailSem19:**
```bash
python tools/test.py \
    configs/domain_adaptation/experiment3_stage3_railsem19.py \
    work_dirs/domain_adaptation/experiment3/stage2_cityscapes/best_mIoU_iter_*.pth
```

---

## View Training Progress in TensorBoard

```bash
tensorboard --logdir work_dirs/domain_adaptation
```

Then open http://localhost:6006 in your browser.
