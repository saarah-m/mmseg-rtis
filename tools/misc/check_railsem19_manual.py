"""Manual check for RailSem19: list images that contain a given class (e.g. truck).

Usage:
  # List val images that contain truck — open these to verify labels or run inference
  python tools/misc/check_railsem19_manual.py --ann-dir data/RailSem19/val/annotations --class-id 14

  # List train images with truck
  python tools/misc/check_railsem19_manual.py --ann-dir data/RailSem19/train/annotations --class-id 14

  # Other rare classes: traffic-light=6, traffic-sign=7, human=11, car=13, rail-embedded=18
  python tools/misc/check_railsem19_manual.py --ann-dir data/RailSem19/val/annotations --class-id 6

To visually check predictions vs GT:
  1) Run test with visualizations saved:
     python tools/test.py CONFIG CHECKPOINT --show-dir work_dirs/vis_val --out work_dirs/preds
  2) Open work_dirs/vis_val/ to see model predictions (colored overlay).
  3) Compare with GT by opening the same filename in data/RailSem19/val/annotations/ in an image viewer.
  4) For truck-specific check, run this script with --class-id 14 and open only those listed images.
"""
import argparse
import os
import os.path as osp
import numpy as np
from PIL import Image

CLASSES = (
    'road', 'sidewalk', 'construction', 'tram-track', 'fence',
    'pole', 'traffic-light', 'traffic-sign', 'vegetation',
    'terrain', 'sky', 'human', 'rail-track', 'car', 'truck',
    'trackbed', 'on-rails', 'rail-raised', 'rail-embedded',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ann-dir', default='data/RailSem19/val/annotations', help='Annotations dir')
    parser.add_argument('--class-id', type=int, default=14, help='Class id to list (14=truck)')
    args = parser.parse_args()

    ann_dir = args.ann_dir
    cid = args.class_id
    name = CLASSES[cid] if cid < len(CLASSES) else str(cid)

    names = sorted(f for f in os.listdir(ann_dir) if f.endswith('.png'))
    with_class = []
    for n in names:
        path = osp.join(ann_dir, n)
        arr = np.array(Image.open(path))
        if np.any(arr == cid):
            with_class.append(n)

    print(f'Class: {name} (id={cid})')
    print(f'Images containing it: {len(with_class)} / {len(names)}')
    for n in with_class[:50]:
        print(' ', n)
    if len(with_class) > 50:
        print(f' ... and {len(with_class) - 50} more')
    print()
    print('To open GT for these: use the filenames above in', ann_dir)
    print('To run predictions: python tools/test.py CONFIG CHECKPOINT --show-dir OUT_DIR')


if __name__ == '__main__':
    main()
