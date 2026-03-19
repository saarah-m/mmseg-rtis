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
        description="Run multi-stage training for Experiment 3: Mapillary → CityScapes → RailSem19"
    )
    parser.add_argument(
        "--work-dir",
        default="work_dirs/domain_adaptation/experiment3_mapillary_cityscapes_to_railsem19",
        help="The base directory to save logs and models for all stages.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in the work_dir of the respective stage.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU ID to use (sets CUDA_VISIBLE_DEVICES)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"Using GPU {args.gpu_id}")

    register_all_modules()

    stages = [
        {
            "name": "Stage 1: Mapillary Pre-training",
            "config": "configs/domain_adaptation/experiment3_stage1_mapillary.py",
            "work_dir": args.work_dir,
        },
        {
            "name": "Stage 2: Cityscapes Fine-tuning",
            "config": "configs/domain_adaptation/experiment3_stage2_cityscapes.py",
            "work_dir": args.work_dir,
        },
        {
            "name": "Stage 3: RailSem19 Fine-tuning",
            "config": "configs/domain_adaptation/experiment3_mapillary_cityscapes_to_railsem19.py",
            "work_dir": args.work_dir,
        },
    ]

    previous_checkpoint = None

    for i, stage in enumerate(stages):
        print(f"\n{'='*80}")
        print(f"🚀 STARTING: {stage['name']}")
        print(f"{'='*80}")

        cfg = Config.fromfile(stage["config"])
        cfg.work_dir = stage["work_dir"]
        cfg.seed = args.seed

        if previous_checkpoint:
            cfg.load_from = previous_checkpoint
        
        # In case of resuming, let the runner handle finding the checkpoint
        cfg.resume = args.resume

        os.makedirs(cfg.work_dir, exist_ok=True)
        cfg.dump(os.path.join(cfg.work_dir, f"config_stage_{i+1}.py"))
        
        # Add stage info to config for logging
        cfg.stage_name = stage['name']
        cfg.stage_number = i + 1

        print(f"Work Directory: {cfg.work_dir}")
        print(f"Config: {stage['config']}")
        print(f"GPU: {args.gpu_id}")
        print(f"Seed: {args.seed}")
        print(f"{'='*80}\n")

        try:
            runner = Runner.from_cfg(cfg)
            runner.train()

            print(f"\n{'='*80}")
            print(f"✅ COMPLETED: {stage['name']}")
            print(f"Results saved to {cfg.work_dir}")
            print(f"{'='*80}")

        except KeyboardInterrupt:
            print(f"\n🛑 Training interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            raise

        # Path to the checkpoint for the next stage
        previous_checkpoint = os.path.join(cfg.work_dir, "latest.pth")

    print(f"\n🎉 All stages of Experiment 3 completed successfully!")
    print(f"All results are in: {args.work_dir}")
    print(f"TensorBoard logs available at: {args.work_dir}/vis_data")


if __name__ == "__main__":
    main()
