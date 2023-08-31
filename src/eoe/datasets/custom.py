import os
import os.path as pt
from collections import Counter
from typing import List, Tuple, Callable, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder, VisionDataset
from torchvision.datasets.folder import default_loader

from eoe.datasets.bases import TorchvisionDataset
from eoe.utils.logger import Logger
from eoe.utils.transformations import ConditionalCompose


class ADCustomDS(TorchvisionDataset):
    base_folder = 'custom'  # appended to root directory as a subdirectory
    ovr = False  # see init doc
    classes = []  # :method:`ADCustomDS.determine_classes` determines this in :method:`eoe.main.init.create_trainer`.

    def __init__(self, root: str, normal_classes: List[int], nominal_label: int,
                 train_transform: transforms.Compose, test_transform: transforms.Compose,
                 raw_shape: Tuple[int, int, int], logger: Logger = None, limit_samples: Union[int, List[int]] = np.infty,
                 train_conditional_transform: ConditionalCompose = None, test_conditional_transform: ConditionalCompose = None,
                 oe=False):
        """
        AD dataset for custom image folder datasets.
        It expects the data being contained in class folders and distinguishes between
        (1) the one-vs-rest (ovr) approach where one class is considered normal
        and is tested against all other classes being anomalous
        (2) the general approach where each class folder contains a normal data folder and an anomalous data folder.
        The :attr:`ovr` determines this.

        For (1) the data folders have to follow this structure:
        root/custom/train/dog/xxx.png
        root/custom/train/dog/xxy.png
        root/custom/train/dog/xxz.png

        root/custom/train/cat/123.png
        root/custom/train/cat/nsdf3.png
        root/custom/train/cat/asd932_.png

        For (2):
        root/custom/train/hazelnut/normal/xxx.png
        root/custom/train/hazelnut/normal/xxy.png
        root/custom/train/hazelnut/normal/xxz.png
        root/custom/train/hazelnut/anomalous/xxa.png    -- may be used during training for OE

        root/custom/train/screw/normal/123.png
        root/custom/train/screw/normal/nsdf3.png
        root/custom/train/screw/anomalous/asd932_.png   -- may be used during training for OE

        The same holds for the test set, where "/train/" has to be replaced by "/test/".

        @param root: Defines the root directory for this dataset. The dataset always its own subdirectory named 'custom',
            (e.g., eoe/data/datasets/custom/ for root being eoe/data/datasets/).
        @param normal_classes: A list of normal classes. Normal training samples are all from these classes.
            Samples from other classes are not available during training. During testing, other classes will be anomalous.
            For (2), the general approach, 'normal_classes' should always contain just one class.
        @param nominal_label: The integer defining the normal (==nominal) label. Usually 0.
        @param train_transform: Preprocessing pipeline used for training, includes all kinds of image transformations.
            May contain the dummy transformation 'norm' that will be replaced with a torchvision normalization instance.
            The required mean and std of the normal training data will be extracted automatically.
        @param test_transform: Preprocessing pipeline used for testing,
            includes all kinds of image transformations but no data augmentation.
            May contain the dummy transformation 'norm' that will be replaced with a torchvision normalization instance.
            The required mean and std of the normal training data will be extracted automatically.
        @param raw_shape: The raw shape of the dataset samples before preprocessing is applied, shape: c x h x w.
        @param logger: Optional. Some logger instance. Is only required for logging warnings related to the datasets.
        @param limit_samples: Optional. If given, limits the number of different samples. That is, instead of using the
            complete dataset, creates a subset that is to be used. If `limit_samples` is an integer, samples a random subset
            with the provided size. If `limit_samples` is a list of integers, create a subset with the indices provided.
        @param train_conditional_transform: Optional. Similar to `train_transform` but conditioned on the label.
            See :class:`eoe.utils.transformations.ConditionalCompose`.
        @param test_conditional_transform: Optional. Similar to `test_transform` but conditioned on the label.
            See :class:`eoe.utils.transformations.ConditionalCompose`.
        @param oe: whether this dataset is used as OE in combination with the normal dataset also being custom.
            In this case the anomalies from the {CLASS}/train/anomalous folder are
            used for the training set instead of the usual {CLASS}/train/normal samples.
            Does not work with the normal custom dataset being in ovr mode.
        """
        root = pt.join(root, self.base_folder)
        super().__init__(
            root, normal_classes, nominal_label, train_transform, test_transform, len(ADCustomDS.classes),
            raw_shape, logger, limit_samples, train_conditional_transform, test_conditional_transform
        )
        self.check_data()
        if self.ovr and oe:
            raise ValueError('Custom datasets in the one-vs-rest mode are mutually exclusive with custom OE.')

        self._train_set = CustomDS(
            self.root, split='train', transform=self.train_transform, target_transform=self.target_transform,
            conditional_transform=self.train_conditional_transform, logger=logger, ovr=self.ovr, nominal_label=nominal_label
        )
        if self.ovr:
            self._train_set = self.create_subset(self._train_set, self._train_set.targets, self._train_set.anomaly_labels)
        else:
            self._train_set = self.create_subset(self._train_set, self._train_set.targets, self._train_set.anomaly_labels, oe=oe)
        self._test_set = CustomDS(
            root=self.root, split='test', transform=self.test_transform, target_transform=self.target_transform,
            conditional_transform=self.test_conditional_transform, logger=logger, ovr=self.ovr, nominal_label=nominal_label
        )
        if self.ovr:
            self._test_set = Subset(self._test_set, list(range(len(self._test_set))))  # create improper subset with all indices
        else:
            self._test_set = Subset(
                self._test_set, np.argwhere(np.isin(self._test_set.targets, np.asarray(self.normal_classes))).flatten().tolist()
            )

    def create_subset(self, dataset_split: VisionDataset, class_labels: List[int], anomaly_labels: List[int], oe=False) -> Subset:
        """
        Similar to the superclass' method, creates a Subset instance for the given dataset split.
        The subset will only contain indices for samples belonging to normal classes according to :attr:`self.normal_classes`.
        Also, it checks the anomaly labels to be normal for the class (see :attr:`oe` for details).
        Further, if :attr:`self.limit_samples` is an integer and not None, it will contain a random subset of
        these normal indices so that len(indices) == `self.limit_samples`.

        However, if `self.limit_samples` is a list of integers, it will overwrite the indices to exactly those defined by
        `self.limit_samples`. Note that in this case it is not assured that all indices are still normal because
        `limit_samples` is not checked for that.

        Since this method uses :attr:`self.normal_classes` and :attr:`self.limit_samples`, it should be used only after
        those have been initialized. In other words, invoke this method after the invoking super().__init__(...).

        @param dataset_split: The prepared dataset split.
        @param class_labels: A list of all sample-wise integer class labels
            (i.e., not for 'normal' and 'anomalous' but, e.g., 'airplane', 'car', etc.). The length of this list has
            to equal the size of the dataset.
        @param anomaly_labels: A list of all sample-wise binary anomaly labels, where :attr:`self.nominal_label` defines
            whether 0 or 1 is nominal.
        @param oe: Determines whether this is an OE dataset, in which case we create a subset with anomalies.
            We only use this when the dataset is not in ovr mode. Then, the custom dataset has a "normal" and "anomalous"
            folder for each class. For testing, the "anomalous" folder contains the ground-truth anomalies for the class.
            For training, it contains potential OE samples for the class. They may be anything. It's up to the user.
            These are the ones we put into the subset if :attr:`oe` is True.
            If :attr:`oe` is False, we, as usual, put the normal samples of the class(es)--
            as determined by :attr:`self.normal_classes`--in the subset.
        @return: The subset containing samples as described above.
        """
        if self.normal_classes is None:
            raise ValueError('Subsets can only be created once the dataset has been initialized.')
        # indices of normal samples according to :attr:`normal_classes`
        normal_idcs = np.argwhere(
            np.isin(np.asarray(class_labels), self.normal_classes) *
            (np.asarray(anomaly_labels) == (self.nominal_label if not oe else (1 - self.nominal_label)))
        ).flatten().tolist()
        if isinstance(self.limit_samples, (int, float)) and self.limit_samples < np.infty:
            # sample randomly s.t. len(normal_idcs) == :attr:`limit_samples`
            normal_idcs = sorted(np.random.choice(normal_idcs, min(self.limit_samples, len(normal_idcs)), False))
        elif not isinstance(self.limit_samples, (int, float)):
            # set indices to :attr:`limit_samples`, note that these are not necessarily normal anymore
            normal_idcs = self.limit_samples
        return Subset(dataset_split, normal_idcs)

    def _get_raw_train_set(self):
        train_set = CustomDS(
            self.root, split='train',
            transform=transforms.Compose(
                [transforms.Resize((self.raw_shape[-1], self.raw_shape[-1])), transforms.ToTensor(), ]),
            target_transform=self.target_transform, logger=self.logger, ovr=self.ovr, nominal_label=self.nominal_label
        )
        return Subset(
            train_set,
            np.argwhere(
                np.isin(np.asarray(train_set.targets), self.normal_classes)
            ).flatten().tolist()
        )

    def n_normal_anomalous(self, train=True) -> dict:
        """
        Extract the number of normal and anomalous data samples.
        @param train: Whether to consider training or test samples.
        @return: A dictionary like {0: #normal_samples, 1: #anomalous_samples} (may change depending on the nominal label)
        """
        ds = self.train_set if train else self.test_set
        return dict(Counter([t for t in np.asarray(ds.dataset.anomaly_labels)[list(set(ds.indices))]]))

    @staticmethod
    def determine_classes(root: str) -> List[str]:
        root = pt.join(root, ADCustomDS.base_folder, 'train')
        classes = sorted([fd for fd in os.listdir(root) if pt.isdir(pt.join(root, fd))])
        ADCustomDS.classes = classes
        return classes

    def check_data(self):
        # custom data check
        trainpath = pt.join(self.root, 'train')
        testpath = pt.join(self.root, 'test')
        if not pt.exists(trainpath):
            raise ValueError(f'No custom data found since {trainpath} does not exist.')
        if not pt.exists(testpath):
            raise ValueError(f'No custom data found since {testpath} does not exist.')
        if self.ovr:
            if any([cls_dir.lower() in ('normal', 'nominal', 'anomalous') for cls_dir in os.listdir(trainpath)]):
                raise ValueError(
                    f'Found a class folder being named "normal", "nominal", or "anomalous" in ({trainpath}). '
                    f'Note that the class folders needs to match the class names (like "dog", "hazelnut"). '
                    f'Deactivate the one-vs-rest mode or change the class folders to class names.'
                )
        else:
            if any([cls_dir.lower() in ('normal', 'nominal', 'anomalous') for cls_dir in os.listdir(trainpath)]):
                raise ValueError(
                    f'Found a class folder being named "normal", "nominal", or "anomalous" in ({trainpath}). '
                    f'Note that the class folders needs to match the class names (like "dog", "hazelnut"). '
                    f'Normal samples need to be placed in CLASS_NAME/normal and anomalous samples in CLASS_NAME/anomalous. '
                )
            for split_dir in (trainpath, testpath):
                for cls_dir in os.listdir(split_dir):
                    if 'normal' not in [d.lower() for d in os.listdir(pt.join(split_dir, cls_dir))]:
                        raise ValueError(
                            f'All class folders need to contain a folder named "normal" for normal samples. '
                            f'However, did not find such a folder in {pt.join(split_dir, cls_dir)}.'
                        )
                    for lbl_dir in os.listdir(pt.join(split_dir, cls_dir)):
                        if lbl_dir.lower() not in ('normal', 'nominal', 'anomalous'):
                            raise ValueError(
                                f'All class folders need to contain folders for "normal" and "anomalous" data. '
                                f'However, found a folder named {lbl_dir} in {pt.join(split_dir, cls_dir)}.'
                            )
        train_classes = os.listdir(trainpath)
        test_classes = os.listdir(testpath)
        if train_classes != test_classes:
            raise ValueError(
                f'The training class names and test class names do no match. '
                f'The training class names are {train_classes} and the test class names {test_classes}.'
            )


class CustomDS(ImageFolder):
    def __init__(self, root: str, split: str = 'train', transform: Callable = None, target_transform: Callable = None,
                 conditional_transform: ConditionalCompose = None, download=True, ovr=False, **kwargs):
        """
        Implements a torchvision-style vision dataset.

        @param root: Root directory where dataset is found or downloaded to.
        @param split: Defines whether to use training or test data. Needs to be in {'train', 'test'}.
        @param transform: A preprocessing pipeline of image transformations.
        @param target_transform: A preprocessing pipeline of label (integer) transformations.
        @param conditional_transform: Optional. A preprocessing pipeline of conditional image transformations.
            See :class:`eoe.datasets.bases.TorchvisionDataset`. Usually this is None.
        @param download: Whether to automatically download this dataset if not found at `root`.
        @param ovr: whether the anomaly labels depend on classes (ovr is True) or on the subfolders' names.
        @param kwargs: Further unimportant optional arguments (e.g., logger).
        @param ovr: See :class:`ADCustomDS`.
        """
        self.logger = kwargs.pop('logger', None)
        self.nominal_label = kwargs.pop('nominal_label', 0)
        self.train = split == 'train'
        self.split = split
        self.ovr = ovr
        assert self.split in ('train', 'test'), f"Split is {self.split} but is required to be in ('train', 'test')."
        root = pt.join(root, self.split)
        super().__init__(root, transform=transform, target_transform=target_transform, **kwargs)
        self.root = root

        self.loader = kwargs.get('loader', default_loader)

        self.conditional_transform = conditional_transform
        self.pre_transform, self.post_transform = None, None
        if self.transform is not None and self.conditional_transform is not None:
            # splits transform at ToTensor(); apply pre_transform - conditional_transform - post_transform (including ToTensor())
            totensor_pos = [isinstance(t, transforms.ToTensor) for t in self.transform.transforms]
            totensor_pos = totensor_pos.index(True) if True in totensor_pos else 0
            self.pre_transform = transforms.Compose(self.transform.transforms[:totensor_pos])
            self.post_transform = transforms.Compose(self.transform.transforms[totensor_pos:])

        if ovr:
            self.anomaly_labels = [self.target_transform(t) for t in self.targets]
        else:
            self.anomaly_labels = [
                self.nominal_label if f.split(os.sep)[-2].lower() in ['normal', 'nominal'] else (1 - self.nominal_label)
                for f, _ in self.samples
            ]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        path, _ = self.samples[index]
        target = self.anomaly_labels[index]

        if self.target_transform is not None:
            pass  # already applied since we use self.anomaly_labels instead of self.targets

        img = self.loader(path)

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

    def __len__(self):
        return len(self.targets)

