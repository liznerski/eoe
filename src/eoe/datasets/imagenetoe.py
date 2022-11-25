import os.path as pt
import sys
from typing import List, Tuple, Callable, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torchvision.datasets.folder import DatasetFolder, default_loader, IMG_EXTENSIONS
from torchvision.datasets.imagenet import ImageNet
from torchvision.datasets.imagenet import verify_str_arg, load_meta_file

from eoe.datasets.bases import TorchvisionDataset
from eoe.utils.logger import Logger
from eoe.utils.transformations import ConditionalCompose


class ADImageNetOE(TorchvisionDataset):
    base_folder = 'imagenet'  # appended to root directory as a subdirectory

    def __init__(self, root: str, normal_classes: List[int], nominal_label: int,
                 train_transform: transforms.Compose, test_transform: transforms.Compose, 
                 raw_shape: Tuple[int, int, int], logger: Logger = None, limit_samples: Union[int, List[int]] = np.infty,
                 train_conditional_transform: ConditionalCompose = None, test_conditional_transform: ConditionalCompose = None):
        """
        AD dataset for ImageNet-1k. Implements :class:`eoe.datasets.bases.TorchvisionDataset`.
        Since we use ImageNet-30 (see :class:`eoe.datasets.imagenet.ADImageNet`) for AD benchmarks, this dataset is merely
        meant to be used as a surrogate in case users don't want to use the cumbersome ImageNet22k dataset as OE.

        This dataset doesn't provide an automatic download. The data needs to be downloaded from https://www.image-net.org/
        and placed in `root`/imagenet/.
        """
        root = pt.join(root, self.base_folder)
        super().__init__(
            root, normal_classes, nominal_label, train_transform, test_transform, 30, raw_shape, logger, limit_samples,
            train_conditional_transform, test_conditional_transform
        )

        self._train_set = MyImageNetOE(
            self.root, split='train', transform=self.train_transform, target_transform=self.target_transform,
            conditional_transform=self.train_conditional_transform, logger=logger
        )
        self._train_set = self.create_subset(self._train_set, self._train_set.targets)
        self._test_set = MyImageNetOE(
            root=self.root, split='val', transform=self.test_transform, target_transform=self.target_transform,
            conditional_transform=self.test_conditional_transform, logger=logger
        )
        self._test_set = Subset(self._test_set, list(range(len(self._test_set))))  # create improper subset with all indices

    def _get_raw_train_set(self):
        train_set = MyImageNetOE(
            self.root, split='train',
            transform=transforms.Compose([transforms.Resize(self.raw_shape[-1]), transforms.CenterCrop(224), transforms.ToTensor(), ]),
            target_transform=self.target_transform, logger=self.logger
        )
        return Subset(
            train_set,
            np.argwhere(
                np.isin(np.asarray(train_set.targets), self.normal_classes)
            ).flatten().tolist()
        )


class MyImageNetOE(ImageNet):
    cache = {'train': {}, 'val': {}}

    def __init__(self, root: str, split: str = 'train', transform: Callable = None, target_transform: Callable = None,
                 conditional_transform: ConditionalCompose = None, **kwargs):
        """
        Reimplements torchvision's ImageNet s.t. it handles the optional conditional transforms and caching of file paths.
        See :class:`eoe.datasets.bases.TorchvisionDataset` for conditional transforms.
        Also, returns (img, target, index) in __get_item__ instead of (img, target).

        Creating a list of all image file paths can take some time for the full ImageNet-1k dataset, which is why
        we simply cache this in RAM (see :attr:`MyImageNetOE.cache`) once loaded at the start of the training so that we
        don't need to repeat this at the start of training each new class-seed combination
        (see :method:`eoe.training.ad_trainer.ADTrainer.run`).
        """
        self.logger = kwargs.pop('logger', None)
        super(DatasetFolder, self).__init__(root, transform=transform, target_transform=target_transform, **kwargs)
        self.split = verify_str_arg(split, "split", ("train", "val"))

        # ------ used cached file paths and other metadata or load and cache them if not available yet
        if len(self.cache[self.split]) == 0:
            print('Load ImageNet meta and cache it...')
            self.parse_archives()
            wnid_to_classes = load_meta_file(self.root)[0]

            self.classes, self.class_to_idx = self.find_classes(self.split_folder)
            self.imgs = self.samples = self.make_dataset(
                self.split_folder, self.class_to_idx,
                kwargs.get('extensions', IMG_EXTENSIONS if kwargs.get('is_valid_file', None) is None else None),
                kwargs.get('is_valid_file', None)
            )
            self.targets = [s[1] for s in self.samples]

            self.wnids = self.classes
            self.wnid_to_idx = self.class_to_idx
            self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
            self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}

            self.cache[self.split] = {}
            self.cache[self.split]['classes'] = self.classes
            self.cache[self.split]['class_to_idx'] = self.class_to_idx
            self.cache[self.split]['samples'] = self.samples
            self.cache[self.split]['targets'] = self.targets
            if self.logger is not None:
                size = sys.getsizeof(self.cache[self.split]['samples']) + sys.getsizeof(self.cache[self.split]['targets'])
                size += sys.getsizeof(self.cache[self.split]['classes']) + sys.getsizeof(self.cache[self.split]['class_to_idx'])
                self.logger.logtxt(
                    f"Cache size of {str(type(self)).split('.')[-1][:-2]}'s meta for split {self.split} is {size * 1e-9:0.3f} GB"
                )
        else:
            print('Use cached ImageNet meta.')
            self.classes = self.cache[self.split]['classes']
            self.class_to_idx = self.cache[self.split]['class_to_idx']
            self.imgs = self.samples = self.cache[self.split]['samples']
            self.targets = self.cache[self.split]['targets']

        self.loader = kwargs.get('loader', default_loader)
        self.extensions = kwargs.get('extensions', None)
        # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------

        self.conditional_transform = conditional_transform
        self.pre_transform, self.post_transform = None, None
        if self.transform is not None and self.conditional_transform is not None:
            # splits transform at ToTensor(); apply pre_transform - conditional_transform - post_transform (including ToTensor())
            totensor_pos = [isinstance(t, transforms.ToTensor) for t in self.transform.transforms]
            totensor_pos = totensor_pos.index(True) if True in totensor_pos else 0
            self.pre_transform = transforms.Compose(self.transform.transforms[:totensor_pos])
            self.post_transform = transforms.Compose(self.transform.transforms[totensor_pos:])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        path, target = self.samples[index]
        img = self.loader(path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            if self.conditional_transform is not None:
                img = self.pre_transform(img)
                img = self.conditional_transform(img, target)
                img = self.post_transform(img)
            else:
                img = self.transform(img)
        if img.std() < 1e-15:  # random crop might yield completely white images (in case of nail)
            img, target, index = self[index]
        return img, target, index


