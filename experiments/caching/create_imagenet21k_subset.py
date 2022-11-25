import argparse
import os
import os.path as pt
import sys
import numpy as np
from torchvision.transforms import Resize, Compose
from tqdm import tqdm
from sre_constants import error as sre_constants_error
from PIL import UnidentifiedImageError

from eoe.datasets import load_dataset, no_classes

parser = argparse.ArgumentParser(
    description="Create ImageNet21k subset by sampling up to N images per class."
)
parser.add_argument(
    '-r', '--resize', type=int, metavar="PX", default=256,
    help='Resize the images of the subset so that the smaller edge matches this. '
)
parser.add_argument('-n', type=int, default=2)
parser.add_argument('-d', '--dst', type=str, metavar='PATH', default=None, help='where to save the subset to')
args = parser.parse_args()

dsstr = "imagenet21k"
split = "train"
datapath = pt.abspath(pt.join(__file__, '../..', '..', 'data', 'datasets'))
if args.dst is None:
    args.dst = pt.join(__file__, '../..', '..', 'data', 'datasets')
dst = pt.abspath(args.dst)
resize = Resize(args.resize)

ds = load_dataset(dsstr, datapath, list(range(no_classes(dsstr))), 0, Compose([]), Compose([]), )
ds = ds.train_set.dataset
mem, n, m = 0, 0, 0
samples = {}
try:
    total = len(ds.imgs)
    procbar = tqdm(total=total, desc=f'---- Sampling {dsstr} subset (current subset size is {n: >8d})')
    i = 0
    while i < total:
        skip = 1
        img_path, _ = ds.imgs[i]
        cls = img_path.split(os.sep)[-2]
        if cls not in samples:
            samples[cls] = [img_path]
            n += 1
        elif len(samples[cls]) < args.n:
            samples[cls].append(img_path)
            n += 1
        else:
            skip = len(os.listdir(pt.dirname(img_path))) - len(samples[cls])
        i += skip
        procbar.update(skip)
        procbar.set_description(f'---- Sampling {dsstr} subset (current subset size is {n: >8d})')
    dst = pt.join(dst, f'{dsstr}_subset_n{n}_r{args.resize}')

    procbar = tqdm(samples.items(), desc=f'---- Storing {dsstr} subset (current subset size is {mem: >9.4f} GB)')
    for cls, img_paths in procbar:
        os.makedirs(pt.join(dst, cls, ))
        for img_path in img_paths:
            try:
                img = resize(ds.loader(img_path))
            except UnidentifiedImageError as e:
                print('Skipping {}. ImageNet22k could not load picture. Unidentified image error.'.format(img_path),
                      file=sys.stderr)
                continue
            except OSError as e:
                print('Skipping {}. ImageNet22k could not load picture. OS error.'.format(img_path), file=sys.stderr)
                continue
            except sre_constants_error as e:
                print('Skipping {}. ImageNet22k could not load picture. sre_constants error.'.format(img_path), file=sys.stderr)
                continue
            mem += np.asarray(img).nbytes * 1e-9  # img shm in GB
            m += 1
            img.save(pt.join(dst, cls, pt.basename(img_path)))

        procbar.set_description(f'---- Storing {dsstr} subset (current subset size is {mem: >9.4f} GB)')

    print(f'---- Created {dsstr} subset at {dst} with {m} samples.')
except KeyboardInterrupt:
    print(f'---- KeyboardInterrupt. Created {dsstr} subset at {dst} with only {m} of {n} samples.')
