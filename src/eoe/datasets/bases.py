import json
import os.path as pt
from abc import ABC, abstractmethod
from collections import Counter
from copy import deepcopy
from typing import Tuple, List, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.data.sampler import RandomSampler
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose
from tqdm import tqdm

from eoe.utils.logger import Logger
from eoe.utils.stats import RunningStats
from eoe.utils.transformations import ConditionalCompose
from eoe.utils.transformations import GPU_TRANSFORMS, Normalize, GlobalContrastNormalization

GCN_NORM = 1
STD_NORM = 0
NORM_MODES = {  # the different transformation dummies that will be automatically replaced by torchvision normalization instances
    'norm': STD_NORM, 'normalise': STD_NORM, 'normalize': STD_NORM,
    'gcn-norm': GCN_NORM, 'gcn-normalize': GCN_NORM, 'gcn-normalise': GCN_NORM,
}


def global_contrast_normalization(x: torch.tensor, scale='l2'):
    """
    Applies global contrast normalization to a tensor; i.e., subtracts the mean across features (pixels) and normalizes by scale,
    which is either the standard deviation, L1- or L2-norm across features (pixels).
    Note that this is a *per sample* normalization globally across features (and not across the dataset).
    """
    assert scale in ('l1', 'l2')
    n_features = int(np.prod(x.shape))
    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean
    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))
    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features
    x /= x_scale
    return x


class BaseADDataset(ABC):
    def __init__(self, root: str):
        """
        An abstract Anomaly Detection (AD) dataset. All AD datasets have a _train_set and a _test_set split that need
        to be prepared during their __init__. They also have a list of normal and anomaly classes.
        @param root: Defines the root directory for all datasets. Most of them get automatically downloaded if not present
            at this directory. Each dataset has its own subdirectory (e.g., eoe/data/datasets/imagenet/).
        """
        super().__init__()
        self.root: str = root  # root path to data

        self.n_classes: int = 2  # 0: normal, 1: outlier
        self.normal_classes: List[int] = None  # tuple with original class labels that define the normal class
        self.outlier_classes: List[int] = None  # tuple with original class labels that define the outlier class

        self._train_set: torch.utils.data.Subset = None  # the actual dataset for training data
        self._test_set: torch.utils.data.Subset = None  # the actual dataset for test data

        self.shape: Tuple[int, int, int] = None  # shape of datapoints, c x h x w
        self.raw_shape: Tuple[int, int, int] = None  # shape of datapoint before preprocessing is applied, c x h x w

    @abstractmethod
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False,
                num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
        """ Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set. """
        pass

    def __repr__(self):
        return self.__class__.__name__


class TorchvisionDataset(BaseADDataset):
    def __init__(self, root: str, normal_classes: List[int], nominal_label: int, train_transform: Compose,
                 test_transform: Compose, classes: int, raw_shape: Tuple[int, int, int],
                 logger: Logger = None, limit_samples: Union[int, List[int]] = np.infty,
                 train_conditional_transform: ConditionalCompose = None, test_conditional_transform: ConditionalCompose = None):
        """
        An implementation of a Torchvision-style AD dataset. It provides a data loader for its train and test split each.
        There is a :method:`preview` that returns a collection of random batches of image samples from the loaders.

        TorchvisionDataset optimizes the transformation pipelines.
        It replaces normalization dummy strings (see :attr:`NORM_MODES`) with actual torchvision normalization instances for which
        it automatically extracts the empirical mean and std of the normal training data and caches it for later use.
        It also moves some transformations automatically to the GPU, for which it removes them from the pipeline
        and stores them in a separate attribute for later use in the ADTrainer (see :class:`eoe.training.ad_trainer.ADTrainer`).

        Implementations of TorchvisionDataset need to create the actual train and test dataset
        (i.e., self._train_set and self._test_set). They also need to create suitable subsets if `limit_samples` is not None.
        Note that self._train_set and self._test_set should always be instances of :class:`torch.utils.data.Subset` even if
        `limit_samples` is None. The training subset still needs to be set so that it excludes all anomalous
        training samples, and even if `normal_classes` contains all classes, the subset simply won't be a proper subset;
        i.e., the Subset instance will have all indices of the complete dataset.
        There is :method:`TorchvisionDataset.create_subset` that can be used for all this.

        @param root: Defines the root directory for all datasets. Most of them get automatically downloaded if not present
            at this directory. Each dataset has its own subdirectory (e.g., eoe/data/datasets/imagenet/).
        @param normal_classes: A list of normal classes. Normal training samples are all from these classes.
            Samples from other classes are not available during training. During testing, other classes will be anomalous.
        @param nominal_label: The integer defining the normal (==nominal) label. Usually 0.
        @param train_transform: Preprocessing pipeline used for training, includes all kinds of image transformations.
            May contain the dummy transformation 'norm' that will be replaced with a torchvision normalization instance.
            The required mean and std of the normal training data will be extracted automatically.
        @param test_transform: Preprocessing pipeline used for testing,
            includes all kinds of image transformations but no data augmentation.
            May contain the dummy transformation 'norm' that will be replaced with a torchvision normalization instance.
            The required mean and std of the normal training data will be extracted automatically.
        @param classes: The number of overall raw classes of this dataset. Static per dataset.
        @param raw_shape: The raw shape of the dataset samples before preprocessing is applied, shape: c x h x w.
        @param logger: Optional. Some logger instance. Is only required for logging warnings related to the datasets.
        @param limit_samples: Optional. If given, limits the number of different samples. That is, instead of using the
            complete dataset, creates a subset that is to be used. If `limit_samples` is an integer, samples a random subset
            with the provided size. If `limit_samples` is a list of integers, create a subset with the indices provided.
        @param train_conditional_transform: Optional. Similar to `train_transform` but conditioned on the label.
            See :class:`eoe.utils.transformations.ConditionalCompose`.
        @param test_conditional_transform: Optional. Similar to `test_transform` but conditioned on the label.
            See :class:`eoe.utils.transformations.ConditionalCompose`.
        """
        super().__init__(root)

        self.raw_shape = raw_shape
        self.normal_classes = tuple(normal_classes)
        normal_set = set(self.normal_classes)
        self.outlier_classes = [c for c in range(classes) if c not in normal_set]
        assert nominal_label in [0, 1]
        self.nominal_label = nominal_label
        self.anomalous_label = 1 if self.nominal_label == 0 else 0
        self.logger = logger
        self.limit_samples = limit_samples

        self.target_transform = transforms.Lambda(
            lambda x: self.anomalous_label if x in self.outlier_classes else self.nominal_label
        )
        self.train_transform = deepcopy(train_transform)
        self.test_transform = deepcopy(test_transform)
        self.gpu_train_transform = lambda x: x
        self.gpu_test_transform = lambda x: x
        self.train_conditional_transform = deepcopy(train_conditional_transform)
        self.test_conditional_transform = deepcopy(test_conditional_transform)
        self.gpu_train_conditional_transform = lambda x, y: x
        self.gpu_test_conditional_transform = lambda x, y: x

        self._unpack_transforms()
        if any([isinstance(t, str) for t in (self.train_transform.transforms + self.test_transform.transforms)]):
            self._update_transforms(self._get_raw_train_set())
            self._unpack_transforms()
        self._split_transforms()

    @property
    def train_set(self):
        return self._train_set

    @property
    def test_set(self):
        return self._test_set

    def create_subset(self, dataset_split: VisionDataset, class_labels: List[int], ) -> Subset:
        """
        Creates a Subset instance for the given dataset split.
        The subset will only contain indices for samples belonging to normal classes according to :attr:`self.normal_classes`.
        Further, if :attr:`self.limit_samples` is an integer and not None, it will contain a random subset of
        these normal indices so that len(indices) == `self.limit_samples`.

        However, if `self.limit_samples` is a list of integers, it will overwrite the indices to exactly those defined by
        `self.limit_samples`. Note that in this case it is not assured that all indices are still normal because
        `limit_samples` is not checked for that.

        Since this method uses :attr:`self.normal_classes` and :attr:`self.limit_samples`, it should be used only after
        those have been initialized. In other words, invoke this method after the implementation of TorchvisionDataset
        invoked super().__init__(...).

        @param dataset_split: The prepared dataset split (e.g., CIFAR-100).
        @param class_labels: A list of all sample-wise integer class labels
            (i.e., not for 'normal' and 'anomalous' but, e.g., 'airplane', 'car', etc.). The length of this list has
            to equal the size of the dataset.
        @return: The subset containing samples as described above.
        """
        if self.normal_classes is None:
            raise ValueError('Subsets can only be created once the dataset has been initialized.')
        # indices of normal samples according to :attr:`normal_classes`
        normal_idcs = np.argwhere(
            np.isin(np.asarray(class_labels), self.normal_classes)
        ).flatten().tolist()
        if isinstance(self.limit_samples, (int, float)) and self.limit_samples < np.infty:
            # sample randomly s.t. len(normal_idcs) == :attr:`limit_samples`
            normal_idcs = sorted(np.random.choice(normal_idcs, min(self.limit_samples, len(normal_idcs)), False))
        elif not isinstance(self.limit_samples, (int, float)):
            # set indices to :attr:`limit_samples`, note that these are not necessarily normal anymore
            normal_idcs = self.limit_samples
        return Subset(dataset_split, normal_idcs)

    def n_normal_anomalous(self, train=True) -> dict:
        """
        Extract the number of normal and anomalous data samples.
        @param train: Whether to consider training or test samples.
        @return: A dictionary like {0: #normal_samples, 1: #anomalous_samples} (may change depending on the nominal label)
        """
        ds = self.train_set if train else self.test_set
        return dict(Counter([self.target_transform(t) for t in np.asarray(ds.dataset.targets)[list(set(ds.indices))]]))

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, replacement=False,
                num_workers: int = 0, persistent=False, prefetch_factor: int = 2) -> Tuple[DataLoader, DataLoader]:
        """
        Returns torch loaders for the train and test split of the dataset.
        @param batch_size: the batch size for the loaders.
        @param shuffle_train: whether to shuffle the train split at the start of each iteration of the data loader.
        @param shuffle_test: whether to shuffle the test split at the start of each iteration of the data loader.
        @param replacement: whether to sample data with replacement.
        @param num_workers: See :class:`torch.util.data.dataloader.DataLoader`.
        @param persistent: See :class:`torch.util.data.dataloader.DataLoader`.
        @param prefetch_factor: See :class:`torch.util.data.dataloader.DataLoader`.
        @return: A tuple (train_loader, test_loader).
        """
        # classes = None means all classes
        train_loader = DataLoader(
            dataset=self.train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=False,
            persistent_workers=persistent, prefetch_factor=prefetch_factor,
            sampler=RandomSampler(self.train_set, replacement=replacement) if shuffle_train else None
        )
        test_loader = DataLoader(
            dataset=self.test_set, batch_size=batch_size, num_workers=num_workers, pin_memory=False,
            persistent_workers=persistent, prefetch_factor=prefetch_factor,
            sampler=RandomSampler(self.test_set, replacement=replacement) if shuffle_test else None
        )
        return train_loader, test_loader

    def preview(self, percls=40, train=True, classes=(0, 1)) -> torch.Tensor:
        """
        Generates a preview of the dataset; i.e., generates a figure of some randomly chosen outputs of the dataloader.
        Therefore, the data samples have already passed the complete preprocessing pipeline.

        @param percls: How many samples (per label) are shown.
        @param train: Whether to show training samples or test samples.
        @param classes: The labels for which images are shown. Defaults to (0, 1) for normal and anomalous.
        @return: A Tensor of images (n x c x h x w).
        """
        if train:
            loader, _ = self.loaders(10, num_workers=0, shuffle_train=True)
        else:
            _, loader = self.loaders(10, num_workers=0, shuffle_test=False)
        x, y, out = torch.FloatTensor(), torch.LongTensor(), []
        for xb, yb, _ in loader:
            xb = xb.cuda()
            if train:
                if isinstance(self, CombinedDataset):
                    xb[yb == self.nominal_label] = self.normal.gpu_train_conditional_transform(
                        xb[yb == self.nominal_label], [self.nominal_label] * len(xb[yb == self.nominal_label])
                    )
                    xb[yb == self.nominal_label] = self.normal.gpu_train_transform(xb[yb == self.nominal_label])
                    xb[yb != self.nominal_label] = self.oe.gpu_train_conditional_transform(
                        xb[yb != self.nominal_label], [self.anomalous_label] * len(xb[yb != self.nominal_label])
                    )
                    xb[yb != self.nominal_label] = self.oe.gpu_train_transform(xb[yb != self.nominal_label])
                else:
                    xb = self.gpu_train_conditional_transform(xb, yb)
                    xb = self.gpu_train_transform(xb)
            else:
                if isinstance(self, CombinedDataset):
                    xb = self.normal.gpu_test_conditional_transform(xb, yb)
                    xb = self.normal.gpu_test_transform(xb)
                else:
                    xb = self.gpu_test_conditional_transform(xb, yb)
                    xb = self.gpu_test_transform(xb)
            xb = xb.cpu()
            x, y = torch.cat([x, xb]), torch.cat([y, yb])
            if all([x[y == c].size(0) >= percls for c in classes]):
                break
        for c in sorted(set(y.tolist())):
            out.append(x[y == c][:percls])
        percls = min(percls, *[o.size(0) for o in out])
        out = [o[:percls] for o in out]
        return torch.cat(out)

    def _update_transforms(self, train_dataset: torch.utils.data.Dataset):
        """
        Replaces occurrences of the string 'Normalize' (or others, see :attr:`NORM_MODES`) within the train and test transforms
        with an actual `transforms.Normalize`. For this, extracts, e.g., the empirical mean and std of the normal data.
        Other transformations might require different statistics, but they will always be used as a mean and std in
        `transforms.Normalize`. For instance, GCN uses a max/min normalization, which can also be accomplished with
        `transforms.Normalize`.
        @param train_dataset: some raw training split of a dataset. In this context, raw means no data augmentation.
        """
        if any([isinstance(t, str) for t in (self.train_transform.transforms + self.test_transform.transforms)]):
            train_str_pos, train_str = list(
                zip(*[(i, t.lower()) for i, t in enumerate(self.train_transform.transforms) if isinstance(t, str)])
            )
            test_str_pos, test_str = list(
                zip(*[(i, t.lower()) for i, t in enumerate(self.test_transform.transforms) if isinstance(t, str)])
            )
            strs = train_str + test_str
            if len(strs) > 0:
                if not all([s in NORM_MODES.keys() for s in strs]):
                    raise ValueError(
                        f'Transforms for dataset contain a string that is not recognized. '
                        f'The only valid strings are {NORM_MODES.keys()}.'
                    )
                if not all([NORM_MODES[strs[i]] == NORM_MODES[strs[j]] for i in range(len(strs)) for j in range(i)]):
                    raise ValueError(f'Transforms contain different norm modes, which is not supported. ')
                if NORM_MODES[strs[0]] == STD_NORM:
                    if self.load_cached_stats(NORM_MODES[strs[0]]) is not None:
                        self.logger.print(f'Use cached mean/std of training dataset with normal classes {self.normal_classes}')
                        mean, std = self.load_cached_stats(NORM_MODES[strs[0]])
                    else:
                        loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True, )
                        acc = RunningStats()
                        desc = f'Extracting mean/std of training dataset with normal classes {self.normal_classes}'
                        for x, _, _ in tqdm(loader, desc=desc):
                            acc.add(x.permute(1, 0, 2, 3).flatten(1).permute(1, 0))
                        mean, std = acc.mean(), acc.std()
                        self.cache_stats(mean, std, NORM_MODES[strs[0]])
                    norm = transforms.Normalize(mean, std, inplace=False)
                else:
                    loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
                    all_x = []
                    for x, _, _ in tqdm(loader, desc='Extracting max and min of GCN normalized training dataset'):
                        x = torch.stack([global_contrast_normalization(xi, scale='l1') for xi in x])
                        all_x.append(x)
                    all_x = torch.cat(all_x)
                    tmin, tmax = all_x.min().item(), all_x.max().item()
                    norm = transforms.Compose([
                        GlobalContrastNormalization(scale='l1'),
                        transforms.Normalize([tmin] * all_x.size(1), [tmax - tmin] * all_x.size(1), inplace=False)
                    ])
                for i in train_str_pos:
                    self.train_transform.transforms[i] = norm
                for i in test_str_pos:
                    self.test_transform.transforms[i] = norm

    def load_cached_stats(self, norm_mode: int) -> Tuple[torch.Tensor, torch.Tensor]:  # returns mean and std of dataset
        """
        Tries to load cached statistics of the normal dataset. :method:`_update_transforms` will first try to use the cache
        before trying to compute the statistics again.
        @param norm_mode: The norm_mode for which the statistics are to be loaded.
        @return: The "mean" and "std" for the corresponding norm_mode (see :attr:`NORM_MODES`)
        """
        file = pt.join(self.root, 'stats_cache.json')
        if pt.exists(file):
            with open(file, 'r') as reader:
                cache = json.load(reader)
            if str(type(self)) in cache and str(norm_mode) in cache[(str(type(self)))] \
                    and json.dumps(self.normal_classes) in cache[str(type(self))][str(norm_mode)]:
                mean, std = cache[str(type(self))][str(norm_mode)][json.dumps(self.normal_classes)]
                return torch.tensor(mean), torch.tensor(std)
        return None

    def cache_stats(self, mean: torch.Tensor, std: torch.Tensor, norm_mode: int):  # caches mean and std of dataset
        """
        Caches the "mean" and "std" for some norm_mode (see :attr:`NORM_MODES`). Is used by :method:`_update_transforms`.
        @param mean: the "mean" (might actually be some other statistic but will be used as a mean for `transforms.Normalize`).
        @param std: the "std" (might actually be some other statistic but will be used as a std for `transforms.Normalize`).
        @param norm_mode: the norm_mode for which the "mean" and "std" are cached.
        """
        file = pt.join(self.root, 'stats_cache.json')
        if not pt.exists(file):
            with open(file, 'w') as writer:
                json.dump({str(type(self)): {str(norm_mode): {}}}, writer)
        with open(file, 'r') as reader:
            cache = json.load(reader)
        if str(type(self)) not in cache:
            cache[str(type(self))] = {}
        if str(norm_mode) not in cache[(str(type(self)))]:
            cache[(str(type(self)))][str(norm_mode)] = {}
        cache[(str(type(self)))][str(norm_mode)][json.dumps(self.normal_classes)] = (mean.numpy().tolist(), std.numpy().tolist())
        with open(file, 'w') as writer:
            json.dump(cache, writer)

    def _split_transforms(self):
        """
        This moves some parts of the preprocessing pipelines (self.train_transform, self.test_transform, etc.)
        to a GPU pipeline. That is, for instance, self.gpu_train_transform. The method automatically looks for transformations
        that appear in :attr:`eoe.utils.transformations.GPU_TRANSFORMS` and replaces them with corresponding GPU versions.
        The :class:`eoe.training.ad_trainer.ADTrainer` accesses self.gpu_train_transform and the other gpu pipelines and
        applies them right after having retrieved the tensors from the dataloader and putting them to the GPU.
        """
        gpu_trans, n = [], 0
        for i, t in enumerate(deepcopy(self.train_transform.transforms)):
            if type(t) in GPU_TRANSFORMS:
                gpu_trans.append(GPU_TRANSFORMS[type(t)](t))
                del self.train_transform.transforms[i-n]
                n += 1
            elif n > 0 and not isinstance(t, transforms.ToTensor):
                raise ValueError('A CPU-only transform follows a GPU transform. This is not supported atm.')
        self.gpu_train_transform = Compose(gpu_trans)
        if not all([isinstance(t, (Normalize, GlobalContrastNormalization)) for t in gpu_trans]):
            raise ValueError(f'Since gpu_train_conditional_transform is applied before gpu_train_transform, '
                             f'gpu_train_transform is not allowed to contain transforms other than Normalize. '
                             f'Otherwise the conditional transforms that are used for the multiscale experiments would be '
                             f'influenced by multiscale generating augmentations.')

        gpu_trans, n = [], 0
        for i, t in enumerate(deepcopy(self.test_transform.transforms)):
            if type(t) in GPU_TRANSFORMS:
                gpu_trans.append(GPU_TRANSFORMS[type(t)](t))
                del self.test_transform.transforms[i-n]
                n += 1
            elif n > 0 and not isinstance(t, transforms.ToTensor):
                raise ValueError('A CPU-only transform follows a GPU transform. This is not supported atm.')
        self.gpu_test_transform = Compose(gpu_trans)
        if not all([isinstance(t, (Normalize, GlobalContrastNormalization)) for t in gpu_trans]):
            raise ValueError(f'Since gpu_test_conditional_transform is applied before gpu_test_transform, '
                             f'gpu_test_transform is not allowed to contain transforms other than Normalize. '
                             f'Otherwise the conditional transforms that are used for the multiscale experiments would be '
                             f'influenced by multiscale generating augmentations.')

        gpu_trans, n = [], 0
        for i, (cond, t1, t2) in enumerate(deepcopy(self.train_conditional_transform.conditional_transforms)):
            if type(t1) in GPU_TRANSFORMS and type(t2) in GPU_TRANSFORMS:
                gpu_trans.append((cond, GPU_TRANSFORMS[type(t1)](t1), GPU_TRANSFORMS[type(t2)](t2)))
                del self.train_conditional_transform.conditional_transforms[i-n]
                n += 1
            elif n > 0:
                raise ValueError('A CPU-only transform follow a GPU transform. This is not supported atm.')
        self.gpu_train_conditional_transform = ConditionalCompose(gpu_trans, gpu=True)

        gpu_trans, n = [], 0
        for i, (cond, t1, t2) in enumerate(deepcopy(self.test_conditional_transform.conditional_transforms)):
            if type(t1) in GPU_TRANSFORMS and type(t2) in GPU_TRANSFORMS:
                gpu_trans.append((cond, GPU_TRANSFORMS[type(t1)](t1), GPU_TRANSFORMS[type(t2)](t2)))
                del self.test_conditional_transform.conditional_transforms[i-n]
                n += 1
            elif n > 0:
                raise ValueError('A CPU-only transform follow a GPU transform. This is not supported atm.')
        self.gpu_test_conditional_transform = ConditionalCompose(gpu_trans, gpu=True)

    def _unpack_transforms(self):
        """ This "unpacks" preprocessing pipelines so that there is Compose inside of a Compose """
        def unpack(t):
            if not isinstance(t, Compose):
                return [t]
            elif isinstance(t, Compose):
                return [tt for t in t.transforms for tt in unpack(t)]
        self.train_transform.transforms = unpack(self.train_transform)
        self.test_transform.transforms = unpack(self.test_transform)

        if self.train_conditional_transform is None:
            self.train_conditional_transform = ConditionalCompose([])
        for cond, t1, t2 in self.train_conditional_transform.conditional_transforms:
            assert not isinstance(t1, Compose) and not isinstance(t2, Compose), 'No Compose inside a ConditionalCompose allowed!'
        if self.test_conditional_transform is None:
            self.test_conditional_transform = ConditionalCompose([])
        for cond, t1, t2 in self.test_conditional_transform.conditional_transforms:
            assert not isinstance(t1, Compose) and not isinstance(t2, Compose), 'No Compose inside a ConditionalCompose allowed!'

    @abstractmethod
    def _get_raw_train_set(self):
        """
        Implement this to return a training set with the corresponding normal class that is used for extracting the mean and std.
        See :method:`_update_transforms`.
        """
        raise NotImplementedError()


class CombinedDataset(TorchvisionDataset):
    def __init__(self, normal_ds: TorchvisionDataset, oe_ds: TorchvisionDataset):
        """
        Creates a combined dataset out of a normal dataset and an Outlier Exposure (OE) dataset.
        The test split and test dataloader of the combined dataset will be the same as the ones of the normal dataset, which has
        both normal and anomalous samples for testing.
        The train split, however, will be a combination of normal training samples and anomalous OE samples.
        For this, it creates a ConcatDataset as a train split.
        More importantly, it creates a :class:`BalancedConcatLoader` for the train split that yields balanced batches
        of equally many normal and OE samples. Note that, the overall returned training batches thus have two times the original
        batch size. If there are not enough OE samples to have equally many different OE samples for the complete normal
        training set, start a new iteration of the OE dataset.
        @param normal_ds: The normal dataset containing only normal training samples but both anomalous and normal test samples.
        @param oe_ds: The Outlier Exposure (OE) dataset containing auxiliary anomalies for training.
        """
        self.normal = normal_ds
        self.oe = oe_ds
        self._train_set = ConcatDataset([self.normal.train_set, self.oe.train_set])
        self._test_set = self.normal.test_set

        self.raw_shape = self.normal.raw_shape
        self.normal_classes = self.normal.normal_classes
        self.outlier_classes = self.normal.outlier_classes
        self.nominal_label = self.normal.nominal_label
        self.anomalous_label = 1 if self.nominal_label == 0 else 0
        self.logger = self.normal.logger
        self.limit_samples = self.oe.limit_samples

    def n_normal_anomalous(self, train=True) -> dict:
        """
        Extract the number of normal and anomalous data samples.
        @param train: Whether to consider training (including OE) or test samples.
        @return: A dictionary like {0: #normal_samples, 1: #anomalous_samples} (may change depending on the nominal label)
        """
        if train:
            normal = self.normal.n_normal_anomalous()
            oe = self.oe.n_normal_anomalous()
            return {k: normal.get(k, 0) + oe.get(k, 0) for k in set.union(set(normal.keys()), set(oe.keys()))}
        else:
            return self.normal.n_normal_anomalous(train)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False,
                num_workers: int = 0, persistent=False, prefetch_factor: int = 2) -> Tuple[DataLoader, DataLoader]:
        """
        Returns the normal datasets' test loader.
        For training, returns a :class:`BalancedConcatLoader` that yields balanced batches
        of equally many normal and OE samples. Note that, the overall returned training batches thus have two times the original
        batch size. If there are not enough OE samples to have equally many different OE samples for the complete normal
        training set, start a new iteration of the OE dataset.
        For a description of the parameters see :method:`eoe.datasets.bases.TorchvisionDataset.loaders`.
        @return: a tuple of (train_loader, test_loader)
        """
        # classes = None means all classes
        normal_train_loader, test_loader = self.normal.loaders(
            batch_size, shuffle_train, shuffle_test, False, num_workers, persistent, prefetch_factor
        )
        oe_train_loader, _ = self.oe.loaders(
            batch_size, shuffle_train, shuffle_test, len(self.oe.train_set.indices) >= 10000, num_workers,
            persistent, prefetch_factor,
        )
        return BalancedConcatLoader(normal_train_loader, oe_train_loader), test_loader

    def _get_raw_train_set(self):
        return None  # doesn't make sense for a combined dataset


class BalancedConcatLoader(object):
    def __init__(self, normal_loader: DataLoader, oe_loader: DataLoader):
        """
        The balanced concat loader samples equally many samples from the normal and oe loader per batch.
        Both types of batches simply get concatenated to form the final batch.
        @param normal_loader: The normal data loader.
        @param oe_loader: The OE data loader.
        """
        self.normal_loader = normal_loader
        self.oe_loader = oe_loader
        if len(self.oe_loader.dataset) < len(self.normal_loader.dataset):
            r = int(np.ceil(len(self.normal_loader.dataset) / len(self.oe_loader.dataset)))
            self.oe_loader.dataset.indices = np.asarray(
                self.oe_loader.dataset.indices
            ).reshape(1, -1).repeat(r, axis=0).reshape(-1).tolist()

    def __iter__(self):
        self.normal_iter = iter(self.normal_loader)
        self.oe_iter = iter(self.oe_loader)
        return self

    def __next__(self):
        normal = next(self.normal_iter)  # imgs, lbls, idxs
        oe = next(self.oe_iter)
        while oe[1].size(0) < normal[1].size(0):
            oe = [torch.cat(a) for a in zip(oe, next(self.oe_iter))]
        oe[-1].data += len(self.normal_loader.dataset.dataset)  # offset indices of OE dataset with normal dataset length
        return [torch.cat([i, j[:i.shape[0]]]) for i, j in zip(normal, oe)]

    def __len__(self):
        return len(self.normal_loader)

