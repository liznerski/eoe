from typing import Tuple, List, Union

import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Subset

from eoe.datasets.bases import TorchvisionDataset
from eoe.utils.logger import Logger
from eoe.utils.transformations import ConditionalCompose


class ADMNIST(TorchvisionDataset):
    def __init__(self, root: str, normal_classes: List[int], nominal_label: int,
                 train_transform: transforms.Compose, test_transform: transforms.Compose, 
                 raw_shape: Tuple[int, int, int], logger: Logger = None, limit_samples: Union[int, List[int]] = np.infty,
                 train_conditional_transform: ConditionalCompose = None, test_conditional_transform: ConditionalCompose = None):
        """ AD dataset for MNIST. Implements :class:`eoe.datasets.bases.TorchvisionDataset`. """
        super().__init__(
            root, normal_classes, nominal_label, train_transform, test_transform, 10, raw_shape, logger, limit_samples,
            train_conditional_transform, test_conditional_transform
        )

        self._train_set = MNIST(
            self.root, train=True, download=True, transform=self.train_transform,
            target_transform=self.target_transform, conditional_transform=self.train_conditional_transform
        )
        self._train_set = self.create_subset(self._train_set, self._train_set.targets)
        self._test_set = MNIST(
            root=self.root, train=False, download=True, transform=self.test_transform,
            target_transform=self.target_transform, conditional_transform=self.test_conditional_transform
        )
        self._test_set = Subset(self._test_set, list(range(len(self._test_set))))  # create improper subset with all indices

    def _get_raw_train_set(self):
        train_set = MNIST(
            self.root, train=True, download=True,
            transform=transforms.Compose([transforms.Resize((self.raw_shape[-1])), transforms.ToTensor(), ]),
            target_transform=self.target_transform
        )
        return Subset(
            train_set,
            np.argwhere(
                np.isin(np.asarray(train_set.targets), self.normal_classes)
            ).flatten().tolist()
        )


class MNIST(torchvision.datasets.MNIST):
    def __init__(self, *args, conditional_transform: ConditionalCompose = None, **kwargs):
        """
        Reimplements torchvision's MNIST s.t. it handles the optional conditional transforms.
        See :class:`eoe.datasets.bases.TorchvisionDataset`. Apart from this, the implementation doesn't differ from the
        standard one.
        """
        super(MNIST, self).__init__(*args, **kwargs)
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
        if self.transform is None or isinstance(self.transform, transforms.Compose) and len(self.transform.transforms) == 0:
            img = img.float().div(255).unsqueeze(0)
        else:
            img = Image.fromarray(img.numpy(), mode="L")
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


class ADEMNIST(TorchvisionDataset):
    def __init__(self, root: str, normal_classes: List[int], nominal_label: int,
                 train_transform: transforms.Compose, test_transform: transforms.Compose,
                 raw_shape: Tuple[int, int, int], logger: Logger = None, limit_samples: Union[int, List[int]] = np.infty,
                 train_conditional_transform: ConditionalCompose = None, test_conditional_transform: ConditionalCompose = None):
        """ AD dataset for EMNIST. Implements :class:`eoe.datasets.bases.TorchvisionDataset`. """
        super().__init__(
            root, normal_classes, nominal_label, train_transform, test_transform, 10, raw_shape, logger, limit_samples,
            train_conditional_transform, test_conditional_transform
        )

        self._train_set = EMNIST(
            self.root, train=True, download=True, transform=self.train_transform,
            target_transform=self.target_transform, conditional_transform=self.train_conditional_transform
        )
        self._train_set = self.create_subset(self._train_set, self._train_set.targets)
        self._test_set = EMNIST(
            root=self.root, train=False, download=True, transform=self.test_transform,
            target_transform=self.target_transform, conditional_transform=self.test_conditional_transform
        )
        self._test_set = Subset(self._test_set, list(range(len(self._test_set))))  # create improper subset with all indices

    def _get_raw_train_set(self):
        train_set = EMNIST(
            self.root, train=True, download=True,
            transform=transforms.Compose([transforms.Resize((self.raw_shape[-1])), transforms.ToTensor(), ]),
            target_transform=self.target_transform
        )
        return Subset(
            train_set,
            np.argwhere(
                np.isin(np.asarray(train_set.targets), self.normal_classes)
            ).flatten().tolist()
        )


class EMNIST(torchvision.datasets.EMNIST):
    def __init__(self, *args, conditional_transform: ConditionalCompose = None, **kwargs):
        """
        Reimplements torchvision's EMNIST s.t. it handles the optional conditional transforms.
        See :class:`eoe.datasets.bases.TorchvisionDataset`.
        Also, returns (img, target, index) in __get_item__ instead of (img, target).
        """
        super(EMNIST, self).__init__(*args, split='letters', **kwargs)
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
        img = img.transpose(0, 1)  # for some reasons, the dimensions are swapped...
        if self.transform is None or isinstance(self.transform, transforms.Compose) and len(self.transform.transforms) == 0:
            img = img.float().div(255).unsqueeze(0)
        else:
            img = Image.fromarray(img.numpy(), mode="L")
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


