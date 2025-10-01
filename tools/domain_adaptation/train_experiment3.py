#!/usr/bin/env python3
"""
Training script for Experiment 3: Mapillary → CityScapes → RailSem19
This is a multi-stage training process.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mmseg.utils import register_all_modules
from mmengine import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Mapillary → CityScapes → RailSem19"
    )
    parser.add_argument(
        "--config",
        default="configs/domain_adaptation/experiment3_mapillary_cityscapes_to_railsem19.py",
        help="config file path",
    )
    parser.add_argument(
        "--work-dir",
        default="work_dirs/domain_adaptation/experiment3_mapillary_cityscapes_to_railsem19",
        help="the directory to save logs and models",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from the latest checkpoint in the work_dir",
    )
    parser.add_argument("--load-from", help="the checkpoint file to load from")
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU ID to use (sets CUDA_VISIBLE_DEVICES)",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="training stage: 1=Mapillary, 2=CityScapes, 3=RailSem19",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"Using GPU {args.gpu_id} (CUDA_VISIBLE_DEVICES={args.gpu_id})")

    # Register all modules
    register_all_modules()

    # Load config
    cfg = Config.fromfile(args.config)

    # Set work directory
    cfg.work_dir = args.work_dir

    # Set random seed
    cfg.seed = args.seed

    # Create work directory
    os.makedirs(args.work_dir, exist_ok=True)

    # Save config to work directory
    cfg.dump(os.path.join(args.work_dir, "config.py"))

    # Build runner
    runner = Runner.from_cfg(cfg)

    # Start training
    runner.train()

    print(f"Training completed! Results saved to {args.work_dir}")


if __name__ == "__main__":
    main()
