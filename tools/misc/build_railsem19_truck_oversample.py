"""Create train_truck_only/ with symlinks to train images that contain truck (label 14).

Run once from repo root. Then use configs that use railsem19_oversample_truck dataset
so truck images are seen ~2x per epoch.

  python tools/misc/build_railsem19_truck_oversample.py
"""
import os
import sys

import numpy as np
from PIL import Image


def main():
    data_root = 'data/RailSem19'
    ann_dir = os.path.join(data_root, 'train', 'annotations')
    out_img_dir = os.path.join(data_root, 'train_truck_only', 'images')
    out_ann_dir = os.path.join(data_root, 'train_truck_only', 'annotations')

    if not os.path.isdir(ann_dir):
        print(f'Not found: {ann_dir}')
        sys.exit(1)

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_ann_dir, exist_ok=True)

    truck_count = 0
    for name in sorted(os.listdir(ann_dir)):
        if not name.endswith('.png'):
            continue
        path = os.path.join(ann_dir, name)
        arr = np.array(Image.open(path))
        if not np.any(arr == 14):
            continue
        truck_count += 1
        base = name[:-4]
        # Symlink image (.jpg) and annotation (.png)
        src_img = os.path.join(data_root, 'train', 'images', base + '.jpg')
        src_ann = os.path.join(data_root, 'train', 'annotations', base + '.png')
        dst_img = os.path.join(out_img_dir, base + '.jpg')
        dst_ann = os.path.join(out_ann_dir, base + '.png')
        for src, dst in [(src_img, dst_img), (src_ann, dst_ann)]:
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)

    print(f'Created train_truck_only with {truck_count} images (symlinks).')
    print(f'Paths: {out_img_dir}, {out_ann_dir}')


if __name__ == '__main__':
    main()
