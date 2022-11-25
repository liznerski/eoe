import os.path as pt
import argparse
from multiprocessing import shared_memory

from tqdm import tqdm
from torchvision.transforms import Compose

from eoe.datasets import load_dataset, no_classes

parser = argparse.ArgumentParser(description="Clear shared memory cache of a dataset. ")
parser.add_argument('dataset', type=str, choices=['imagenet30', 'cub', 'dtd', 'imagenet21ksubset'])
parser.add_argument('-s', '--split', type=str, choices=['train', 'test'], default='train')
args = parser.parse_args()

dsstr = args.dataset
dsstr = dsstr if dsstr != 'imagenet30' else 'imagenet'
split = args.split
split = split if (split == 'train' or dsstr != 'imagenet') else 'val'
datapath = pt.abspath(pt.join(__file__, '../..', '..', 'data', 'datasets'))

ds = load_dataset(dsstr, datapath, list(range(no_classes(dsstr))), 0, Compose([]), Compose([]), )
if split == 'train':
    ds = ds.train_set.dataset
else:
    ds = ds.test_set.dataset
shared_imgs = []
cleared = 0
procbar = tqdm(ds.imgs, desc=f'---- Clearing cache {dsstr}')

for i, img in enumerate(procbar):
    try:
        shm = shared_memory.SharedMemory(name=f'{dsstr}_{split}_{i}')
        shm.close()
        shm.unlink()
        cleared += 1
    except FileNotFoundError:
        pass

print(f'Cleared {cleared} shared_memory objects. The others seemed to be cleared already.')
