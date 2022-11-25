import os
import os.path as pt
import sys
from multiprocessing import shared_memory
from typing import List, Tuple, Callable, Union

import PIL.Image
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from multiprocessing.resource_tracker import unregister  # careful with this!
from torch.utils.data import Subset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms.functional import to_pil_image

from eoe.datasets.bases import TorchvisionDataset
from eoe.utils.caching import decode_shape_and_image
from eoe.utils.logger import Logger
from eoe.utils.transformations import ConditionalCompose


class ADCUB(TorchvisionDataset):
    base_folder = 'cub'  # appended to root directory as a subdirectory

    def __init__(self, root: str, normal_classes: List[int], nominal_label: int,
                 train_transform: transforms.Compose, test_transform: transforms.Compose,
                 raw_shape: Tuple[int, int, int], logger: Logger = None, limit_samples: Union[int, List[int]] = np.infty,
                 train_conditional_transform: ConditionalCompose = None, test_conditional_transform: ConditionalCompose = None):
        """
        AD dataset for CUB-200-2011 (https://www.vision.caltech.edu/datasets/cub_200_2011/).
        Implements :class:`eoe.datasets.bases.TorchvisionDataset`.
        """
        root = pt.join(root, self.base_folder)
        super().__init__(
            root, normal_classes, nominal_label, train_transform, test_transform, 200, raw_shape, logger, limit_samples,
            train_conditional_transform, test_conditional_transform
        )

        self._train_set = CUB(
            self.root, split='train', transform=self.train_transform, target_transform=self.target_transform,
            conditional_transform=self.train_conditional_transform, logger=logger
        )
        self._train_set = self.create_subset(self._train_set, self._train_set.targets)
        self._test_set = CUB(
            root=self.root, split='test', transform=self.test_transform, target_transform=self.target_transform,
            conditional_transform=self.test_conditional_transform, logger=logger
        )
        self._test_set = Subset(self._test_set, list(range(len(self._test_set))))  # create improper subset with all indices

    def _get_raw_train_set(self):
        train_set = CUB(
            self.root, split='train',
            transform=transforms.Compose(
                [transforms.Resize(self.raw_shape[-1]), transforms.CenterCrop(224), transforms.ToTensor(), ]),
            target_transform=self.target_transform, logger=self.logger
        )
        return Subset(
            train_set,
            np.argwhere(
                np.isin(np.asarray(train_set.targets), self.normal_classes)
            ).flatten().tolist()
        )


class CUB(VisionDataset):
    # adapted from https://github.com/TDeVries/cub2011_dataset/blob/master/cub2011.py
    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    base_folder = 'CUB_200_2011'  # appended to root directory as a subdirectory

    def __init__(self, root: str, split: str = 'train', transform: Callable = None, target_transform: Callable = None,
                 conditional_transform: ConditionalCompose = None, download=True, **kwargs):
        """
        Implements a torchvision-style vision dataset.
        This implementation also uses shared memory if prepared by other scripts (see experiments/caching folder).
        Loading data from shared memory speeds up data loading a lot if multiple experiments using CUB run in parallel
        on the same machine. However, using shared memory can cause memory leaks, which is why we don't recommend using it.
        CUB automatically falls back to loading the data from disk as usual if a sample is not found in shared memory.

        @param root: Root directory where dataset is found or downloaded to.
        @param split: Defines whether to use training or test data. Needs to be in {'train', 'test'}.
        @param transform: A preprocessing pipeline of image transformations.
        @param target_transform: A preprocessing pipeline of label (integer) transformations.
        @param conditional_transform: Optional. A preprocessing pipeline of conditional image transformations.
            See :class:`eoe.datasets.bases.TorchvisionDataset`. Usually this is None.
        @param download: Whether to automatically download this dataset if not found at `root`.
        @param kwargs: Further unimportant optional arguments (e.g., logger).
        """
        self.logger = kwargs.pop('logger', None)
        super().__init__(root, transform=transform, target_transform=target_transform, **kwargs)
        self.root = root
        self.train = split == 'train'
        self.split = split
        if download:
            self._download()

        self.loader = kwargs.get('loader', default_loader)

        self.conditional_transform = conditional_transform
        self.pre_transform, self.post_transform = None, None
        if self.transform is not None and self.conditional_transform is not None:
            # splits transform at ToTensor(); apply pre_transform - conditional_transform - post_transform (including ToTensor())
            totensor_pos = [isinstance(t, transforms.ToTensor) for t in self.transform.transforms]
            totensor_pos = totensor_pos.index(True) if True in totensor_pos else 0
            self.pre_transform = transforms.Compose(self.transform.transforms[:totensor_pos])
            self.post_transform = transforms.Compose(self.transform.transforms[totensor_pos:])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        path, target = self.samples[index], self.targets[index]

        try:  # try to use shared memory if available (see experiments/caching folder)
            img = self._load_image_from_shared_memory(index)
            # self.logger.logtxt(f'{self.split: >5}: Used shm for {index}', prnt=False)
        except FileNotFoundError:  # Shared memory (cached CUB) not found, load from disk
            img = self.loader(path)
            # self.logger.logtxt(f'{self.split: >5}: Disk load for {index}', prnt=False)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            if self.conditional_transform is not None:
                img = self.pre_transform(img)
                img = self.conditional_transform(img, target)
                img = self.post_transform(img)
            else:
                img = self.transform(img)
        if img.std() < 1e-15:  # random crop might yield completely white images, retry
            img, target, index = self[index]
        return img, target, index

    def _load_image_from_shared_memory(self, index: int) -> PIL.Image.Image:
        shm = shared_memory.SharedMemory(name=f'cub_{self.split}_{index}')
        img = decode_shape_and_image(np.copy(np.ndarray(shm.size, dtype=np.uint8, buffer=shm.buf)))
        shm.close()
        # The following line is a fix to make shared_memory work with unrelated processes.
        # Shared memory can cause memory leaks (indefinitely persisting memory allocations)
        # if the resources are not properly released/cleaned-up.
        # Python makes sure to prevent this leak by automatically starting a hidden resource_tracker process that
        # outlives the parent. It terminates and cleans up any shared memory (i.e., releases it) once
        # the parent dies. Unfortunately, each unrelated python process starts its own resource_tracker, which is
        # unaware of the other trackers and processes. This results in the tracker releasing all shared memory instances that
        # the parent process has been linked to--even the ones that it read but didn't write--once the parent terminates.
        # Other unrelated processes--e.g., the one that created the shared memory instance or other processes
        # that still want to read the data--can thus not access the data anymore since it has been released already.
        # One solution would be to make sure that all the processes use the same resource_tracker.
        # However, this would require to have one mother process that starts all experiments on the machine, which
        # would be very annoying in practice (e.g., one would have to wait until all processes are finished until new
        # experiments can be started that use the shared memory).
        # The other solution is presented below. Since the datasets only read and never write shared memory, we
        # can quite safely tell the resource_tracker that "we are going to deal with the clean-up manually" by unregistering
        # the read shared memory from this resource_tracker.
        # It is, however, very important to not do this with the shared-memory-creating process since this could cause
        # memory leaks as at least one process must release the resources!!!
        # See https://stackoverflow.com/questions/64102502/shared-memory-deleted-at-exit.
        unregister(shm._name, 'shared_memory')
        img = to_pil_image(img)
        return img

    def __len__(self):
        return len(self.targets)

    def _load_metadata(self) -> bool:
        if not pt.exists(os.path.join(self.root, self.base_folder, 'images.txt')):
            return False

        images = pd.read_csv(os.path.join(self.root, self.base_folder, 'images.txt'), sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(
            os.path.join(self.root, self.base_folder, 'image_class_labels.txt'), sep=' ', names=['img_id', 'target']
        )
        train_test_split = pd.read_csv(
            os.path.join(self.root, self.base_folder, 'train_test_split.txt'), sep=' ', names=['img_id', 'is_training_img']
        )

        data = images.merge(image_class_labels, on='img_id')
        data = data.merge(train_test_split, on='img_id')

        if self.train:
            data = data[data.is_training_img == 1]
        else:
            data = data[data.is_training_img == 0]

        data['filepath'] = pt.join(self.root, self.base_folder, 'images') + os.sep + data['filepath']
        self.imgs = self.samples = data.filepath.values
        self.targets = data.target.values - 1  # (1, ..., 200) -> (0, ..., 199)
        self.classes = [f.split('.')[-1] for f in sorted([f for f in set([f.split(os.sep)[-2] for f in self.imgs])])]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        return True

    def _check_integrity(self):
        ret = self._load_metadata()
        if not ret:
            return False

        for fp in self.samples:
            if not pt.isfile(fp):
                print(fp, 'is not found.', file=sys.stderr)
                return False
        return True

    def _download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        else:
            download_and_extract_archive(self.url, pt.join(self.root, ))
            assert self._check_integrity(), 'CUB is corrupted. Please redownload.'
