import argparse
import os.path as pt
import sys
from multiprocessing import shared_memory
from threading import Event
from datetime import datetime, timedelta
from time import time

import numpy as np
from torchvision.transforms import Resize, Compose
from tqdm import tqdm

from eoe.datasets import load_dataset, no_classes
from eoe.utils.caching import encode_shape_and_image

parser = argparse.ArgumentParser(
    description="Cache a dataset to the main memory using shared memory. "
                "Some of the datasets automatically look for data samples in the shared memory before trying to "
                "load them from the disk. This speeds up data loading. "
                "Be careful with this cript, killing it violently might cause memory leaks on the system!"
)
parser.add_argument('dataset', type=str, choices=['imagenet30', 'cub', 'dtd', 'imagenet21ksubset'])
parser.add_argument('-s', '--split', type=str, choices=['train', 'test'], default='train')
parser.add_argument('-l', '--limit', type=int, metavar='GB', default=30, help='Only cache up to this many GB.',)
parser.add_argument(
    '-r', '--resize', type=int, metavar="PX", default=256,
    help='Resize the images so that the smaller edge matches this. Cache the resized images.'
)
parser.add_argument(
    '-t', '--time', type=float, metavar="DAYS", required=True,
    help='Defines the days after which the caching automatically terminates and frees the allocated shared memory.'
)
args = parser.parse_args()

start = time()
dsstr = args.dataset
dsstr = dsstr if dsstr != 'imagenet30' else 'imagenet'
split = args.split
split = split if (split == 'train' or dsstr != 'imagenet') else 'val'
datapath = pt.abspath(pt.join(__file__, '../..', '..', 'data', 'datasets'))
limit = args.limit  # GB
resize = Resize(args.resize)

ds = load_dataset(dsstr, datapath, list(range(no_classes(dsstr))), 0, Compose([]), Compose([]), )
if split == 'train':
    ds = ds.train_set.dataset
else:
    ds = ds.test_set.dataset
shared_imgs = []
mem = 0
procbar = tqdm(ds.imgs, desc=f'---- Caching {dsstr} (current cache size is {mem: >9.4f} GB)')
try:
    for i, img in enumerate(procbar):
        if isinstance(img, tuple):  # ImageNet returns (img_path, label) instead of (img_path)
            img_path = img[0]
        else:
            img_path = img
        img = np.asarray(resize(ds.loader(img_path)))
        mem += img.nbytes * 1e-9  # img shm in GB
        if mem > limit:
            print(f'---- Memory threshold ({limit} GB) exceeded. '
                  f'Stop at this point. Cached only {len(shared_imgs)}/{len(ds.imgs)} images.')
            break
        shm = shared_memory.SharedMemory(create=True, size=15 + img.nbytes, name=f'{dsstr}_{split}_{i}')
        dst = np.ndarray(shape=(15 + img.nbytes, ), dtype=img.dtype, buffer=shm.buf)
        dst[:] = encode_shape_and_image(img)[:]

        shared_imgs.append(shm)
        procbar.set_description(f'---- Caching {dsstr} (current cache size is {mem: >9.4f} GB)')
    Event().wait(timeout=0.1)

    wait = args.time * 24 * 60 * 60 - (time() - start)  # days -> seconds
    until = datetime.strftime(datetime.now() + timedelta(seconds=wait), "%Y/%m/%d %H:%M:%S")
    print(f'Cached images are available as "{dsstr}_{split}_IDX".')
    print(f'---- Provided shared memory. Wait until {until} or KeyboardInterrupt...')
    Event().wait(timeout=wait)
except KeyboardInterrupt:
    pass
finally:
    print('---- Releasing shared memory...')
    for shm in shared_imgs:
        try:
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            print(f'Shared memory {shm.name} not found. Closed externally?', file=sys.stderr)
    print('---- Shared memory released. Terminate.')
