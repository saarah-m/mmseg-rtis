#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Run inference with best checkpoint for each model on validation images.

Output structure:
    inference_comparison/
        rs19-config.json              <- top-level, shared across all scenes
        rs00033/
            input.jpg                 <- original RGB photo
            gt.png                    <- ground truth (grayscale, pixel = class index 0-18)
            deeplabv3plus.png         <- model prediction (same grayscale format)
            mask2former.png
            ocrnet.png
        rs00034/
            ...

Mask PNG format: grayscale uint8, pixel value = class index (0-18), 255 = ignore.
The viewer colorizes them client-side using rs19-config.json.

Usage:
    python tools/run_all_models_inference.py [--val-dir VAL_DIR] [--out-dir OUT_DIR] [--device DEVICE]
    python tools/run_all_models_inference.py --images rs00161,rs01118,rs03890

Example:
    python tools/run_all_models_inference.py
    python tools/run_all_models_inference.py --images rs00161,rs01118,rs03890,rs01079,rs02356,rs02906,rs03918,rs04114,rs04726,rs05327
    python tools/run_all_models_inference.py --val-dir data/RailSem19/rs19_4000/validation_images --device cpu

python tools/run_all_models_inference.py --images rs00161,rs01118,rs03890,rs01079,rs02356,rs02906,rs03918,rs04114,rs04726,rs05327
THIS COVERS ALL CLASSES

"""

import argparse
import gc
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from mmengine.model import revert_sync_batchnorm
from PIL import Image
from tqdm import tqdm

from mmseg.apis import inference_model, init_model


DEFAULT_VAL_DIR = "data/RailSem19/rs19_4000/validation_images"


def find_gt_mask_path(img_path: Path) -> Optional[Path]:
    """Resolve ground truth mask path by swapping *_images -> *_masks."""
    for split in ("validation_images", "train_images", "test_images"):
        if split in str(img_path):
            mask_path = Path(str(img_path).replace(split, split.replace("_images", "_masks")))
            mask_path = mask_path.with_suffix(".png")
            if mask_path.exists():
                return mask_path
    return None


def get_best_checkpoint(work_dir: Path) -> Optional[Path]:
    """Return best_mIoU_*.pth if present, otherwise the checkpoint from last_checkpoint."""
    best = list(work_dir.glob("best_mIoU_*.pth"))
    if best:
        return best[0]
    last_file = work_dir / "last_checkpoint"
    if last_file.exists():
        ckpt = last_file.read_text().strip()
        for candidate in (work_dir / ckpt, Path(ckpt)):
            if candidate.exists():
                return candidate
    return None


def model_short_name(work_dir_name: str) -> str:
    """e.g. 'deeplabv3plus_r101-d8_4xb2-80k_railsem19-540x960' -> 'deeplabv3plus'"""
    return work_dir_name.split("_")[0]


def save_grayscale_mask(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array.astype(np.uint8), mode="L").save(str(path))


def discover_models(work_dirs: Path) -> List[Tuple[str, Path, Path]]:
    """Return list of (short_name, config_path, checkpoint_path) for valid models."""
    results = []
    for model_dir in sorted(d for d in work_dirs.iterdir() if d.is_dir()):
        config_file = model_dir / f"{model_dir.name}.py"
        if not config_file.exists():
            print(f"  skip {model_dir.name}: no config")
            continue
        checkpoint = get_best_checkpoint(model_dir)
        if checkpoint is None:
            print(f"  skip {model_dir.name}: no checkpoint")
            continue
        short = model_short_name(model_dir.name)
        results.append((short, config_file, checkpoint))
        print(f"  found {short} ({checkpoint.name})")
    return results


def prepare_io(images: List[Path], out_dir: Path) -> None:
    """Copy input images and GT masks (pure I/O, no GPU)."""
    for img_path in tqdm(images, desc="Copying inputs + GTs", unit="img"):
        scene_id = img_path.stem
        scene_dir = out_dir / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(img_path, scene_dir / f"input{img_path.suffix}")

        mask_path = find_gt_mask_path(img_path)
        if mask_path is not None:
            img_size = Image.open(str(img_path)).size  # (W, H)
            mask = Image.open(str(mask_path))
            if mask.mode != "L":
                mask = mask.split()[0]
            if mask.size != img_size:
                mask = mask.resize(img_size, Image.NEAREST)
            mask.save(str(scene_dir / "gt.png"))


def run_model_on_all(
    short_name: str,
    config_path: Path,
    checkpoint_path: Path,
    images: List[Path],
    out_dir: Path,
    device: str,
) -> None:
    """Load one model, run all images, unload. Keeps only one model in VRAM."""
    print(f"\nLoading {short_name}...")
    model = init_model(str(config_path), str(checkpoint_path), device=device)
    if device == "cpu":
        model = revert_sync_batchnorm(model)
    model.eval()
    use_fp16 = device.startswith("cuda")

    for img_path in tqdm(images, desc=f"  {short_name}", unit="img"):
        scene_dir = out_dir / img_path.stem
        out_file = scene_dir / f"{short_name}.png"
        if out_file.exists():
            continue
        try:
            with torch.cuda.amp.autocast(enabled=use_fp16):
                result = inference_model(model, str(img_path))
            pred = result.pred_sem_seg.data.squeeze().cpu().numpy()
            save_grayscale_mask(pred, out_file)
        except Exception as e:
            tqdm.write(f"  ERROR [{img_path.stem}] {short_name}: {e}")

    # Free GPU memory before loading next model
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on validation images and save per-scene results"
    )
    parser.add_argument(
        "--val-dir",
        default=DEFAULT_VAL_DIR,
        help=f"Validation images directory (default: {DEFAULT_VAL_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        default="inference_comparison",
        help="Root output directory (default: inference_comparison)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for inference (default: cuda:0)",
    )
    parser.add_argument(
        "--work-dirs",
        default="work_dirs",
        help="Path to work_dirs containing model subdirectories (default: work_dirs)",
    )
    parser.add_argument(
        "--images",
        default=None,
        help="Comma-separated list of image stems to process (e.g. rs00161,rs01118). "
             "If omitted, processes all images in --val-dir.",
    )
    args = parser.parse_args()

    val_dir = Path(args.val_dir)
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    if args.images:
        stems = [s.strip() for s in args.images.split(",")]
        images = []
        for stem in stems:
            matches = list(val_dir.glob(f"{stem}.*"))
            if matches:
                images.append(matches[0])
            else:
                print(f"WARNING: image '{stem}' not found in {val_dir}, skipping")
        images = sorted(images)
    else:
        images = sorted(val_dir.glob("*.jpg")) + sorted(val_dir.glob("*.png"))

    if not images:
        raise FileNotFoundError(f"No images found in {val_dir}")

    work_dirs = Path(args.work_dirs)
    if not work_dirs.exists():
        raise FileNotFoundError(f"work_dirs not found: {work_dirs}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- rs19-config.json at top level ----
    config_json = work_dirs.parent / "data" / "RailSem19" / "rs19-config.json"
    if not config_json.exists():
        config_json = val_dir.parent.parent / "rs19-config.json"
    if config_json.exists():
        shutil.copy2(config_json, out_dir / "rs19-config.json")
        print(f"Copied class config -> {out_dir / 'rs19-config.json'}")

    # ---- Discover models ----
    print(f"\nScanning {work_dirs}:")
    model_specs = discover_models(work_dirs)
    if not model_specs:
        print("No models found — nothing to do.")
        return

    print(f"\n{len(model_specs)} model(s), {len(images)} image(s)")

    # ---- Phase 1: I/O (copy inputs + GT masks) ----
    prepare_io(images, out_dir)

    # ---- Phase 2: Inference (one model at a time to maximize GPU utilization) ----
    for short_name, config_path, checkpoint_path in model_specs:
        run_model_on_all(
            short_name, config_path, checkpoint_path,
            images, out_dir, args.device,
        )

    print(f"\nDone. Results in {out_dir.absolute()}")


if __name__ == "__main__":
    main()
