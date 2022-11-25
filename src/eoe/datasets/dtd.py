from multiprocessing import shared_memory
from typing import List, Tuple, Union

import PIL.Image
import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from PIL import Image
from multiprocessing.resource_tracker import unregister  # careful with this!
from torch.utils.data import Subset
from torchvision.transforms.functional import to_pil_image

from eoe.datasets.bases import TorchvisionDataset
from eoe.utils.caching import decode_shape_and_image
from eoe.utils.logger import Logger
from eoe.utils.transformations import ConditionalCompose


class ADDTD(TorchvisionDataset):
    def __init__(self, root: str, normal_classes: List[int], nominal_label: int,
                 train_transform: transforms.Compose, test_transform: transforms.Compose, 
                 raw_shape: Tuple[int, int, int], logger: Logger = None, limit_samples: Union[int, List[int]] = np.infty,
                 train_conditional_transform: ConditionalCompose = None, test_conditional_transform: ConditionalCompose = None):
        """
        AD dataset for DTD (describable textures, https://www.robots.ox.ac.uk/~vgg/data/dtd/).
        Implements :class:`eoe.datasets.bases.TorchvisionDataset`.
        """
        super().__init__(
            root, normal_classes, nominal_label, train_transform, test_transform, 47, raw_shape, logger, limit_samples,
            train_conditional_transform, test_conditional_transform
        )

        self._train_set = DTD(
            self.root, split='train', download=True, transform=self.train_transform,
            target_transform=self.target_transform, conditional_transform=self.train_conditional_transform
        )
        self._train_set = self.create_subset(self._train_set, self._train_set.targets)
        self._test_set = DTD(
            root=self.root, split='test', download=True, transform=self.test_transform,
            target_transform=self.target_transform, conditional_transform=self.test_conditional_transform
        )
        self._test_set = Subset(self._test_set, list(range(len(self._test_set))))  # create improper subset with all indices

    def _get_raw_train_set(self):
        train_set = DTD(
            self.root, split='train', download=True,
            transform=transforms.Compose([transforms.Resize(self.raw_shape[-1]), transforms.CenterCrop(224), transforms.ToTensor(), ]),
            target_transform=self.target_transform
        )
        return Subset(
            train_set,
            np.argwhere(
                np.isin(np.asarray(train_set.targets), self.normal_classes)
            ).flatten().tolist()
        )


class DTD(torchvision.datasets.DTD):
    def loader(self, img_path: str) -> Image.Image:  # default PIL image loader
        return Image.open(img_path).convert("RGB")

    def __init__(self, *args, conditional_transform: ConditionalCompose = None, **kwargs):
        """
        Reimplements torchvision's DTD s.t. it handles the optional conditional transforms.
        See :class:`eoe.datasets.bases.TorchvisionDataset`.
        This implementation also uses shared memory if prepared by other scripts (see experiments/caching folder).
        Loading data from shared memory speeds up data loading a lot if multiple experiments using DTD run in parallel
        on the same machine. However, using shared memory can cause memory leaks, which is why we don't recommend using it.
        DTD automatically falls back to loading the data from disk as usual if a sample is not found in shared memory.
        Also, returns (img, target, index) in __get_item__ instead of (img, target).
        """
        super(DTD, self).__init__(*args, **kwargs)
        self.data = self.samples = self.imgs = self._image_files
        self.targets = self._labels
        self.conditional_transform = conditional_transform
        self.pre_transform, self.post_transform = None, None
        if self.transform is not None and self.conditional_transform is not None:
            # splits transform at ToTensor(); apply pre_transform - conditional_transform - post_transform (including ToTensor())
            totensor_pos = [isinstance(t, transforms.ToTensor) for t in self.transform.transforms]
            totensor_pos = totensor_pos.index(True) if True in totensor_pos else 0
            self.pre_transform = transforms.Compose(self.transform.transforms[:totensor_pos])
            self.post_transform = transforms.Compose(self.transform.transforms[totensor_pos:])

    def __getitem__(self, index) -> Tuple[torch.Tensor, int, int]:
        img, target = self.data[index], self.targets[index]

        try:  # try to use shared memory if available
            img = self._load_image_from_shared_memory(index)
            # self.logger.logtxt(f'{self.split: >5}: Used shm for {index}', prnt=False)
        except FileNotFoundError:  # Shared memory (cached DTD) not found, load from disk
            img = self.loader(img)
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
        return img, target, index

    def _load_image_from_shared_memory(self, index: int) -> PIL.Image.Image:
        shm = shared_memory.SharedMemory(name=f'dtd_{self._split}_{index}')
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

