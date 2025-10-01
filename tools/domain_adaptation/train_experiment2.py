#!/usr/bin/env python3
"""
Training script for Experiment 2: CityScapes → RailSem19
Enhanced with progress monitoring and TensorBoard integration
"""

import argparse
import os
import sys
import subprocess
import threading
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mmseg.utils import register_all_modules
from mmengine import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description="Train CityScapes → RailSem19")
    parser.add_argument(
        "--config",
        default="configs/domain_adaptation/experiment2_cityscapes_to_railsem19.py",
        help="config file path",
    )
    parser.add_argument(
        "--work-dir",
        default="work_dirs/domain_adaptation/experiment2_cityscapes_to_railsem19",
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
    parser.add_argument("--tensorboard", action="store_true", help="Start TensorBoard automatically")
    parser.add_argument("--tensorboard-port", type=int, default=6006, help="TensorBoard port")
    parser.add_argument("--monitor", action="store_true", help="Start progress monitoring")
    parser.add_argument("--monitor-interval", type=int, default=30, help="Monitoring interval in seconds")

    return parser.parse_args()


def start_tensorboard(work_dir, port=6006):
    """Start TensorBoard in a separate process."""
    try:
        # Wait a bit for vis_data directory to be created
        time.sleep(5)
        
        vis_data_dir = Path(work_dir) / "vis_data"
        if vis_data_dir.exists():
            cmd = ["tensorboard", "--logdir", str(vis_data_dir), "--port", str(port)]
            process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"🚀 TensorBoard started at http://localhost:{port}")
            return process
        else:
            print("⚠️  vis_data directory not found, TensorBoard not started")
            return None
    except Exception as e:
        print(f"❌ Failed to start TensorBoard: {e}")
        return None

def start_monitoring(work_dir, interval=30):
    """Start progress monitoring in a separate thread."""
    def monitor_loop():
        from tools.domain_adaptation.monitor_training import TrainingMonitor
        monitor = TrainingMonitor(work_dir, "CityScapes → RailSem19")
        monitor.monitor_continuously(interval)
    
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
    print(f"🔄 Progress monitoring started (interval: {interval}s)")
    return monitor_thread

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

    # Start TensorBoard if requested
    tensorboard_process = None
    if args.tensorboard:
        tensorboard_process = start_tensorboard(args.work_dir, args.tensorboard_port)

    # Start monitoring if requested
    monitor_thread = None
    if args.monitor:
        monitor_thread = start_monitoring(args.work_dir, args.monitor_interval)

    print(f"\n{'='*80}")
    print(f"🚀 STARTING EXPERIMENT 2: CityScapes → RailSem19")
    print(f"{'='*80}")
    print(f"Work Directory: {args.work_dir}")
    print(f"Config: {args.config}")
    print(f"GPU: {args.gpu_id}")
    print(f"Seed: {args.seed}")
    if args.tensorboard:
        print(f"TensorBoard: http://localhost:{args.tensorboard_port}")
    if args.monitor:
        print(f"Progress Monitoring: Active (interval: {args.monitor_interval}s)")
    print(f"{'='*80}\n")

    try:
        # Build runner
        runner = Runner.from_cfg(cfg)

        # Start training
        runner.train()

        print(f"\n{'='*80}")
        print(f"✅ TRAINING COMPLETED: CityScapes → RailSem19")
        print(f"{'='*80}")
        print(f"Results saved to: {args.work_dir}")
        if args.tensorboard:
            print(f"TensorBoard: http://localhost:{args.tensorboard_port}")
        print(f"{'='*80}")

    except KeyboardInterrupt:
        print(f"\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise
    finally:
        # Cleanup
        if tensorboard_process:
            try:
                tensorboard_process.terminate()
                print("🛑 TensorBoard stopped")
            except:
                pass


if __name__ == "__main__":
    main()
