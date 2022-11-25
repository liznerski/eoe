import os.path as pt
from typing import Tuple, List, Callable, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Subset
from torchvision.datasets import VisionDataset

from eoe.datasets.bases import TorchvisionDataset
from eoe.utils.logger import Logger
from eoe.utils.transformations import ConditionalCompose


class ADTinyImages(TorchvisionDataset):
    def __init__(self, root: str, normal_classes: List[int], nominal_label: int,
                 train_transform: transforms.Compose, test_transform: transforms.Compose,
                 raw_shape: Tuple[int, int, int], logger: Logger = None, limit_samples: Union[int, List[int]] = np.infty,
                 train_conditional_transform: ConditionalCompose = None, test_conditional_transform: ConditionalCompose = None):
        """
        AD dataset for 80MTI. Implements :class:`eoe.datasets.bases.TorchvisionDataset`.
        Doesn't use any class labels, and doesn't have a test split. Therefore, this is only suitable to be used as OE.

        ADTinyImages doesn't provide an automatic download. The 80MTI dataset has been withdrawn since it contains offensive
        images (https://groups.csail.mit.edu/vision/TinyImages/). We also encourage to not use this for further research
        but decided to use it for our experiments to be comparable with previous work.
        """
        super().__init__(
            root, normal_classes, nominal_label, train_transform, test_transform, 1, raw_shape, logger, limit_samples,
            train_conditional_transform, test_conditional_transform
        )

        self._train_set = TinyImages(
            self.root, transform=self.train_transform,
            target_transform=self.target_transform, conditional_transform=self.train_conditional_transform
        )
        self._train_set = self.create_subset(self._train_set, self._train_set.targets)

    def _get_raw_train_set(self):
        train_set = TinyImages(
            self.root, transform=transforms.Compose([transforms.Resize((self.raw_shape[-1])), transforms.ToTensor(), ]),
            target_transform=self.target_transform
        )
        return Subset(
            train_set,
            np.argwhere(
                np.isin(np.asarray(train_set.targets), self.normal_classes)
            ).flatten().tolist()
        )


class TinyImages(VisionDataset):
    basefolder = 'tinyimages'  # appended to root directory as a subdirectory
    filename = 'tiny_images.bin'  # file name of the dataset
    cached_non_cifar_idxs = None

    def __init__(self, root: str, *args, transform: Callable = None, target_transform: Callable = None, exclude_cifar=True,
                 conditional_transform: ConditionalCompose = None, **kwargs):
        """
        Implements a torchvision-style TinyImages dataset.
        Caches all non-cifar indices in RAM at :attr:`TinyImages.cached_non_cifar_idx` so that later instances during this run
        do not need to reload the metadata.

        @param root: Root directory for data. The data is just one tiny_images.bin file (~433 GB) placed at `root`/`filename`/.
        @param transform: A preprocessing pipeline of image transformations.
        @param target_transform: A preprocessing pipeline of label (integer) transformations.
        @param exclude_cifar: Whether to exclude the CIFAR-10 and CIFAR-100 classes.
            If true, there needs to be a file '80mn_cifar_idxs.txt' at `root`/tinyimages/ that contains all
            indices of CIFAR-10 and CIFAR-100 in 80MTI line-wise.
        @param conditional_transform: Optional. A preprocessing pipeline of conditional image transformations.
            See :class:`eoe.datasets.bases.TorchvisionDataset`. Usually this is None.
        @param args: see :class:`torchvision.DatasetFolder`.
        @param kwargs: see :class:`torchvision.DatasetFolder`.
        """
        super(TinyImages, self).__init__(
            pt.join(root, self.basefolder), *args, transform=transform, target_transform=target_transform, **kwargs
        )
        self.conditional_transform = conditional_transform
        self.pre_transform, self.post_transform = None, None
        if self.transform is not None and self.conditional_transform is not None:
            # splits transform at ToTensor(); apply pre_transform - conditional_transform - post_transform (including ToTensor())
            totensor_pos = [isinstance(t, transforms.ToTensor) for t in self.transform.transforms]
            totensor_pos = totensor_pos.index(True) if True in totensor_pos else 0
            self.pre_transform = transforms.Compose(self.transform.transforms[:totensor_pos])
            self.post_transform = transforms.Compose(self.transform.transforms[totensor_pos:])

        self.exclude_cifar = exclude_cifar
        self.data_file = open(pt.join(self.root, self.filename), 'rb')
        self.targets = np.zeros(79302016)

        if exclude_cifar:
            self.cifar_idxs = []
            with open(pt.join(self.root, '80mn_cifar_idxs.txt'), 'r') as idxs:
                for idx in idxs:
                    # indices in file take the 80mn database to start at 1, hence "- 1"
                    self.cifar_idxs.append(int(idx) - 1)
            self.cifar_idxs = set(self.cifar_idxs)
            self.in_cifar = lambda x: x in self.cifar_idxs
            if self.cached_non_cifar_idxs is None:
                self.non_cifar_idxs = [i for i in range(len(self)) if not self.in_cifar(i)]
                TinyImages.cached_non_cifar_idxs = self.non_cifar_idxs
            else:
                self.non_cifar_idxs = self.cached_non_cifar_idxs
            self.targets = np.zeros(79302016)[:-len(self.cifar_idxs)]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        target = 0

        if self.target_transform is not None:
            target = self.target_transform(target)

        img = self.load_image(index)
        img = Image.fromarray(img)
        if self.transform is not None:
            if self.conditional_transform is not None:
                img = self.pre_transform(img)
                img = self.conditional_transform(img, target)
                img = self.post_transform(img)
            else:
                img = self.transform(img)

        return img, target, index

    def __len__(self) -> int:
        return len(self.targets)

    def load_image(self, idx):
        if self.exclude_cifar:
            idx = self.non_cifar_idxs[idx]
        self.data_file.seek(idx * 3072)
        data = self.data_file.read(3072)
        npimg = np.frombuffer(data, dtype='uint8').reshape(32, 32, 3, order='F')
        return npimg
