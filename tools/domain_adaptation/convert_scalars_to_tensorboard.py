#!/usr/bin/env python3
"""
Convert scalars.json files to TensorBoard event files.
This is useful when TensorBoard backend wasn't enabled during training.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Try to import TensorBoard SummaryWriter
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        print("Error: Neither torch.utils.tensorboard nor tensorboardX is available.")
        print("Please install one of them:")
        print("  pip install tensorboard")
        print("  pip install tensorboardX")
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert scalars.json to TensorBoard events'
    )
    parser.add_argument(
        'scalars_json',
        help='Path to scalars.json file'
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory for TensorBoard events (default: same as scalars.json dir)'
    )
    parser.add_argument(
        '--log-name',
        default='converted',
        help='Name for the TensorBoard log (default: converted)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    scalars_file = Path(args.scalars_json)
    if not scalars_file.exists():
        print(f"Error: {scalars_file} does not exist!")
        return
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = scalars_file.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading scalars from: {scalars_file}")
    print(f"Output directory: {output_dir}")
    
    # Create TensorBoard writer
    writer = SummaryWriter(str(output_dir))
    
    # Read and process scalars.json line by line
    count = 0
    with open(scalars_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            step = data.get('step', data.get('iter', 0))
            
            # Write all metrics to TensorBoard
            for key, value in data.items():
                if key not in ['step', 'iter'] and isinstance(value, (int, float)):
                    # Create hierarchical tags for better organization
                    if key.startswith('decode.'):
                        tag = f'train/{key}'
                    elif key in ['loss', 'data_time', 'time', 'memory']:
                        tag = f'train/{key}'
                    elif key in ['lr', 'base_lr']:
                        tag = f'learning_rate/{key}'
                    else:
                        tag = f'metrics/{key}'
                    
                    writer.add_scalar(tag, value, step)
            
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} entries...")
    
    writer.close()
    print(f"\n✅ Successfully converted {count} entries!")
    print(f"TensorBoard events saved to: {output_dir}")
    print(f"\nTo view in TensorBoard, run:")
    print(f"  tensorboard --logdir {output_dir.parent}")


if __name__ == '__main__':
    main()

