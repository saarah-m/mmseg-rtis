"""RailSem19: count how many images contain each class.

Usage:
  python tools/misc/railsem19_class_stats.py
"""
import os
import sys

import numpy as np
from PIL import Image

CLASSES = (
    'road', 'sidewalk', 'construction', 'tram-track', 'fence',
    'pole', 'traffic-light', 'traffic-sign', 'vegetation',
    'terrain', 'sky', 'human', 'rail-track', 'car', 'truck',
    'trackbed', 'on-rails', 'rail-raised', 'rail-embedded',
)


def main():
    data_root = os.environ.get('MMSEG_DATA_ROOT', 'data/RailSem19')
    train_dir = os.path.join(data_root, 'train', 'annotations')
    val_dir = os.path.join(data_root, 'val', 'annotations')

    if not os.path.isdir(train_dir):
        print(f'Not found: {train_dir}', file=sys.stderr)
        sys.exit(1)

    def count_per_split(ann_dir):
        names = sorted(f for f in os.listdir(ann_dir) if f.endswith('.png'))
        total = len(names)
        counts = [0] * 19
        for n in names:
            arr = np.array(Image.open(os.path.join(ann_dir, n)))
            for cid in range(19):
                if np.any(arr == cid):
                    counts[cid] += 1
        return total, counts

    train_total, train_counts = count_per_split(train_dir)
    val_total, val_counts = count_per_split(val_dir)

    print('RailSem19: images containing each class')
    print('=' * 72)
    print(f"{'Class':<18} {'Train':>10} {'Val':>10} {'Train %':>10} {'Val %':>10}")
    print('-' * 72)
    for cid in range(19):
        name = CLASSES[cid]
        tc, vc = train_counts[cid], val_counts[cid]
        tp = 100 * tc / train_total
        vp = 100 * vc / val_total
        print(f'{name:<18} {tc:>10} {vc:>10} {tp:>9.1f}% {vp:>9.1f}%')
    print('-' * 72)
    print(f"{'TOTAL images':<18} {train_total:>10} {val_total:>10}")


if __name__ == '__main__':
    main()
