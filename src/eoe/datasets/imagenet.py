import os.path as pt
import sys
from multiprocessing import shared_memory
from sre_constants import error as sre_constants_error
from typing import List, Tuple, Callable, Union

import PIL.Image
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import UnidentifiedImageError
from multiprocessing.resource_tracker import unregister  # careful with this!
from torch.utils.data import Subset
from torchvision.datasets.folder import DatasetFolder, default_loader, IMG_EXTENSIONS, has_file_allowed_extension
from torchvision.datasets.imagenet import ImageFolder
from torchvision.datasets.imagenet import verify_str_arg
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from eoe.datasets.bases import TorchvisionDataset
from eoe.utils.caching import decode_shape_and_image
from eoe.utils.logger import Logger
from eoe.utils.transformations import ConditionalCompose


class ADImageNet(TorchvisionDataset):
    ad_classes = ['acorn', 'airliner', 'ambulance', 'american_alligator', 'banjo', 'barn', 'bikini', 'digital_clock',
                  'dragonfly', 'dumbbell', 'forklift', 'goblet', 'grand_piano', 'hotdog', 'hourglass', 'manhole_cover',
                  'mosque', 'nail', 'parking_meter', 'pillow', 'revolver', 'rotary_dial_telephone', 'schooner',
                  'snowmobile', 'soccer_ball', 'stingray', 'strawberry', 'tank', 'toaster', 'volcano']  # the 30 AD classes
    base_folder = 'imagenet_ad'  # appended to root directory as a subdirectory

    def __init__(self, root: str, normal_classes: List[int], nominal_label: int,
                 train_transform: transforms.Compose, test_transform: transforms.Compose, 
                 raw_shape: Tuple[int, int, int], logger: Logger = None, limit_samples: Union[int, List[int]] = np.infty,
                 train_conditional_transform: ConditionalCompose = None, test_conditional_transform: ConditionalCompose = None):
        """
        AD dataset for ImageNet-30. Following Hendrycks et al. (https://arxiv.org/abs/1812.04606) this AD dataset
        is limited to 30 of the 1000 classes of ImageNet (see :attr:`ADImageNet.ad_classes`). Accordingly, the
        class indices are redefined, ranging from 0 to 29, ordered alphabetically.
        Implements :class:`eoe.datasets.bases.TorchvisionDataset`.

        This dataset doesn't provide an automatic download. The data needs to be either downloaded from
        https://github.com/hendrycks/ss-ood, which already contains only the AD classes, or from https://www.image-net.org/.
        It needs to be placed in `root`/imagenet_ad/.
        """
        root = pt.join(root, self.base_folder)
        super().__init__(
            root, normal_classes, nominal_label, train_transform, test_transform, 30, raw_shape, logger, limit_samples,
            train_conditional_transform, test_conditional_transform
        )

        self._train_set = MyImageNet(
            self.root, split='train', transform=self.train_transform, target_transform=self.target_transform,
            conditional_transform=self.train_conditional_transform, logger=logger
        )
        # The following removes all samples from classes not in ad_classes
        # This shouldn't be necessary if the data from https://github.com/hendrycks/ss-ood is used
        self.train_ad_classes_idx = [self._train_set.class_to_idx[c] for c in self.ad_classes]
        self._train_set.targets = [  # t = nan if not in ad_classes else give id in order of ad_classes
            self.train_ad_classes_idx.index(t) if t in self.train_ad_classes_idx else np.nan for t in self._train_set.targets
        ]
        self._train_set.samples = [(s, tn) for (s, to), tn in zip(self._train_set.samples, self._train_set.targets)]
        # create a subset using only normal samples and limit the variety according to :attr:`limit_samples`
        self._train_set = self.create_subset(self._train_set, self._train_set.targets)

        self._test_set = MyImageNet(
            root=self.root, split='val', transform=self.test_transform, target_transform=self.target_transform,
            conditional_transform=self.test_conditional_transform, logger=logger
        )
        # The following removes all samples from classes not in ad_classes
        # This shouldn't be necessary if the data from https://github.com/hendrycks/ss-ood is used
        self.test_ad_classes_idx = [self._test_set.class_to_idx[c] for c in self.ad_classes]
        self._test_set.targets = [  # t = nan if not in ad_classes else give id in order of ad_classes
            self.test_ad_classes_idx.index(t) if t in self.test_ad_classes_idx else np.nan
            for t in self._test_set.targets
        ]
        self._test_set.samples = [(s, tn) for (s, to), tn in zip(self._test_set.samples, self._test_set.targets)]
        self._test_set = Subset(
            self._test_set,
            np.argwhere(
                np.isin(np.asarray(self._test_set.targets), list(range(len(self.ad_classes))))
            ).flatten().tolist()
        )
        
        assert self.test_ad_classes_idx == self.train_ad_classes_idx

    def _get_raw_train_set(self):
        train_set = MyImageNet(
            self.root, split='train',
            transform=transforms.Compose([transforms.Resize(self.raw_shape[-1]), transforms.CenterCrop(224), transforms.ToTensor(), ]),
            target_transform=self.target_transform, logger=self.logger
        )
        train_ad_classes_idx = [train_set.class_to_idx[c] for c in self.ad_classes]
        train_set.targets = [  # t = nan if not in ad_classes else give id in order of ad_classes
            train_ad_classes_idx.index(t) if t in train_ad_classes_idx else np.nan for t in train_set.targets
        ]
        train_set.samples = [(s, tn) for (s, to), tn in zip(train_set.samples, train_set.targets)]
        return Subset(
            train_set,
            np.argwhere(
                np.isin(np.asarray(train_set.targets), self.normal_classes)
            ).flatten().tolist()
        )


class MyImageNet(ImageFolder):
    cache = {'train': {}, 'val': {}}

    def __init__(self, root: str, split: str = 'train', transform: Callable = None, target_transform: Callable = None,
                 conditional_transform: ConditionalCompose = None, **kwargs):
        """
        Reimplements torchvision's ImageNet s.t. it handles the optional conditional transforms, caching of file paths,
        and shared memory loading. See :class:`eoe.datasets.bases.TorchvisionDataset` for conditional transforms.
        Also, returns (img, target, index) in __get_item__ instead of (img, target).

        Creating a list of all image file paths can take some time for the full ImageNet-1k dataset, which is why
        we simply cache this in RAM (see :attr:`MyImageNet.cache`) once loaded at the start of the training so that we
        don't need to repeat this at the start of training each new class-seed combination
        (see :method:`eoe.training.ad_trainer.ADTrainer.run`).

        This implementation uses shared memory if prepared by other scripts (see experiments/caching folder).
        Loading data from shared memory speeds up data loading a lot if multiple experiments using MyImageNet run in parallel
        on the same machine. However, using shared memory can cause memory leaks, which is why we don't recommend using it.
        MyImageNet automatically falls back to loading the data from disk as usual if a sample is not found in shared memory.
        """
        self.logger = kwargs.pop('logger', None)
        super(DatasetFolder, self).__init__(root, transform=transform, target_transform=target_transform, **kwargs)
        self.split = verify_str_arg(split, "split", ("train", "val"))
        self.split_folder = pt.join(self.root, self.split)

        # ------ used cached file paths and other metadata or load and cache them if not available yet
        if len(self.cache[self.split]) == 0:
            print('Load ImageNet meta and cache it...')

            self.classes, self.class_to_idx = self.find_classes(self.split_folder)
            self.imgs = self.samples = self.make_dataset(
                self.split_folder, self.class_to_idx, is_valid_file=self.is_valid_file,
            )
            self.targets = [s[1] for s in self.samples]

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

        try:  # try to use shared memory if available
            img = self._load_image_from_shared_memory(index)
            # self.logger.logtxt(f'{self.split: >5}: Used shm for {index}', prnt=False)
        except FileNotFoundError:  # Shared memory (cached imagenet) not found, load from disk
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
        if img.std() < 1e-15:  # random crop might yield completely white images (in case of nail)
            img, target, index = self[index]
        return img, target, index

    def _load_image_from_shared_memory(self, index: int) -> PIL.Image.Image:
        shm = shared_memory.SharedMemory(name=f'imagenet_{self.split}_{index}')
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

    def is_valid_file(self, file: str) -> bool:
        # check for file extension and ignore corrupt file in hendrycks' imagenet_30 dataset
        return has_file_allowed_extension(file, IMG_EXTENSIONS) and not file.endswith('airliner/._1.JPEG')


class ADImageNet21k(TorchvisionDataset):
    base_folder = pt.join('imagenet22k', 'fall11_whole_extracted')  # appended to root directory as subdirectories
    img_cache_size = 10000  # cache up to this many MB of images to RAM

    def __init__(self, root: str, normal_classes: List[int], nominal_label: int,
                 train_transform: transforms.Compose, test_transform: transforms.Compose,
                 raw_shape: Tuple[int, int, int], logger: Logger = None, limit_samples: Union[int, List[int]] = np.infty,
                 train_conditional_transform: ConditionalCompose = None, test_conditional_transform: ConditionalCompose = None):
        """
        AD dataset for ImageNet-21k. Implements :class:`eoe.datasets.bases.TorchvisionDataset`.
        Doesn't use any class labels, and doesn't have a test split. Therefore, this is only suitable to be used as OE.

        This implementation also automatically caches some images in RAM if limit_samples is not np.infty.
        It only caches up to ~10 GB of data. The rest will be loaded from disk or shared memory as usual.
        Caching samples in RAM only makes sense for experiment with very limited amount of OE.
        For example, if there are only 2 OE samples, it doesn't make sense to reload them from the disk all the time.
        Note that data augmentation will still be applied on images loaded from RAM.

        ADImageNet21k doesn't provide an automatic download. The data needs to be downloaded from https://www.image-net.org/
        and placed in `root`/imagenet22k/fall11_whole_extracted/.
        """
        root = pt.join(root, self.base_folder)
        super().__init__(
            root, normal_classes, nominal_label, train_transform, test_transform, 21811, raw_shape, logger, limit_samples,
            train_conditional_transform, test_conditional_transform,
        )

        self._train_set = ImageNet22K(
            self.root, transform=self.train_transform, target_transform=self.target_transform, logger=self.logger,
            conditional_transform=self.train_conditional_transform, subset='_subset' in self.base_folder
        )
        normal_idcs = np.argwhere(
            np.isin(np.asarray(self._train_set.targets), self.normal_classes)
        ).flatten().tolist()
        if isinstance(limit_samples, (int, float)) and limit_samples < np.infty:
            normal_idcs = sorted(np.random.choice(normal_idcs, min(limit_samples, len(normal_idcs)), False))
        elif not isinstance(limit_samples, (int, float)):
            normal_idcs = limit_samples
        if limit_samples != np.infty:
            self._train_set.cache(normal_idcs[:ADImageNet21k.img_cache_size])
        self._train_set = Subset(self._train_set, normal_idcs)

    def _get_raw_train_set(self):
        train_set = ImageNet22K(
            self.root, transform=transforms.Compose([
                transforms.Resize(self.raw_shape[-1]), transforms.CenterCrop(224), transforms.ToTensor(),
            ]),
            target_transform=self.target_transform, subset='_subset' in self.base_folder
        )
        return Subset(
            train_set,
            np.argwhere(
                np.isin(np.asarray(train_set.targets), self.normal_classes)
            ).flatten().tolist()
        )


class ImageNet22K(ImageFolder):
    imagenet1k_pairs = [
        ('acorn', 'n12267677'),
        ('airliner', 'n02690373'),
        ('ambulance', 'n02701002'),
        ('american_alligator', 'n01698640'),
        ('banjo', 'n02787622'),
        ('barn', 'n02793495'),
        ('bikini', 'n02837789'),
        ('digital_clock', 'n03196217'),
        ('dragonfly', 'n02268443'),
        ('dumbbell', 'n03255030'),
        ('forklift', 'n03384352'),
        ('goblet', 'n03443371'),
        ('grand_piano', 'n03452741'),
        ('hotdog', 'n07697537'),
        ('hourglass', 'n03544143'),
        ('manhole_cover', 'n03717622'),
        ('mosque', 'n03788195'),
        ('nail', 'n03804744'),
        ('parking_meter', 'n03891332'),
        ('pillow', 'n03938244'),
        ('revolver', 'n04086273'),
        ('rotary_dial_telephone', 'n03187595'),
        ('schooner', 'n04147183'),
        ('snowmobile', 'n04252077'),
        ('soccer_ball', 'n04254680'),
        ('stingray', 'n01498041'),
        ('strawberry', 'n07745940'),
        ('tank', 'n04389033'),
        ('toaster', 'n04442312'),
        ('volcano', 'n09472597')
    ]
    imagenet1k_labels = [label for name, label in imagenet1k_pairs]
    cached_samples = None
    cached_targets = None
    cached_classes = None
    cached_class_to_idx = None

    def __init__(self, root: str, *args, transform: Callable = None, target_transform: Callable = None, logger: Logger = None,
                 exclude_imagenet1k=True, conditional_transform: ConditionalCompose = None, **kwargs):
        """
        Implements a torchvision-style ImageNet22k dataset similar to torchvision's ImageNet implementation.
        Based on torchvision's ImageFolder implementation.
        The data needs to be downloaded manually from https://www.image-net.org/ and put in `root`/.

        Creating a list of all image file paths can take some time for the full ImageNet-22k dataset, which is why
        we simply cache this in RAM (see :attr:`ImageNet22K.cached_samples` etc.) once loaded at the start of the training
        so that we don't need to repeat this at the start of training each new class-seed combination
        (see :method:`eoe.training.ad_trainer.ADTrainer.run`).

        This implementation uses shared memory if prepared by other scripts (see experiments/caching folder).
        Loading data from shared memory speeds up data loading a lot if multiple experiments using ImageNet22k run in parallel
        on the same machine. However, using shared memory can cause memory leaks, which is why we don't recommend using it.
        ImageNet22k automatically falls back to loading the data from disk as usual if a sample is not found in shared memory.

        @param root: Root directory for data.
        @param transform: A preprocessing pipeline of image transformations.
        @param target_transform: A preprocessing pipeline of label (integer) transformations.
        @param logger: Optional logger instance. Only used for logging warnings.
        @param exclude_imagenet1k: Whether to exclude ImageNet-1k classes.
        @param conditional_transform: Optional. A preprocessing pipeline of conditional image transformations.
            See :class:`eoe.datasets.bases.TorchvisionDataset`. Usually this is None.
        @param args: See :class:`torchvision.DatasetFolder`.
        @param kwargs: See :class:`torchvision.DatasetFolder`.
        """
        self.subset = kwargs.pop('subset', False)
        super(DatasetFolder, self).__init__(root, *args, transform=transform, target_transform=target_transform, **kwargs)
        self.logger = logger
        self.conditional_transform = conditional_transform
        self.pre_transform, self.post_transform = None, None
        if self.transform is not None and self.conditional_transform is not None:
            # splits transform at ToTensor(); apply pre_transform - conditional_transform - post_transform (including ToTensor())
            totensor_pos = [isinstance(t, transforms.ToTensor) for t in self.transform.transforms]
            totensor_pos = totensor_pos.index(True) if True in totensor_pos else 0
            self.pre_transform = transforms.Compose(self.transform.transforms[:totensor_pos])
            self.post_transform = transforms.Compose(self.transform.transforms[totensor_pos:])

        # ------ used cached file paths and other metadata or load and cache them if not available yet
        if self.cached_samples is None:
            assert exclude_imagenet1k, 'Caching always excludes imagenet1k'
            print('Load ImageNet21k meta and cache it...')
            self.classes, self.class_to_idx = self.find_classes(self.root)
            self.samples = self.imgs = self.make_dataset(
                self.root, self.class_to_idx, 
                kwargs.get('extensions', IMG_EXTENSIONS if kwargs.get('is_valid_file', None) is None else None), 
                kwargs.get('is_valid_file', None)
            )
            self.targets = [s[1] for s in self.samples]

            if exclude_imagenet1k:
                imagenet1k_idxs = tuple([self.class_to_idx.get(label) for label in self.imagenet1k_labels])
                self.samples = self.imgs = [s for s in self.samples if s[1] not in imagenet1k_idxs]  # s = ('<path>', idx) pair
                self.targets = [s[1] for s in self.samples]
                for label in self.imagenet1k_labels:
                    try:
                        self.classes.remove(label)
                        del self.class_to_idx[label]
                    except:
                        pass

            ImageNet22K.cached_samples = self.samples
            ImageNet22K.cached_targets = self.targets
            ImageNet22K.cached_classes = self.classes
            ImageNet22K.cached_class_to_idx = self.class_to_idx
            if self.logger is not None:
                size = sys.getsizeof(ImageNet22K.cached_samples) + sys.getsizeof(ImageNet22K.cached_targets)
                size += sys.getsizeof(ImageNet22K.cached_classes) + sys.getsizeof(ImageNet22K.cached_class_to_idx)
                self.logger.logtxt(
                    f"Cache size of {str(type(self)).split('.')[-1][:-2]}'s meta is {size * 1e-9:0.3f} GB"
                )
        else:
            print('Use cached ImageNet21k meta.')
            self.samples = self.imgs = self.cached_samples
            self.targets = self.cached_targets
            self.classes = self.cached_classes
            self.class_to_idx = self.cached_class_to_idx

        self.loader = kwargs.get('loader', default_loader)
        self.extensions = kwargs.get('extensions', None)
        # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------

        self.exclude_imagenet1k = exclude_imagenet1k
        self.cached_images = {}

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        """
        Override the original method of the ImageFolder class to catch some errors.
        For example, it seems like some ImageNet22k images are broken. If this occurs, just sample the next index.
        Further, this implementation supports conditional transforms and shared memory loading.
        Also, returns (img, target, index) instead of (img, target).
        """
        path, target = self.samples[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        try:
            if self.load_cached(index) is not None:
                sample = self.load_cached(index)
            else:
                try:  # try to use shared memory if available
                    sample = self._load_image_from_shared_memory(index)
                    # self.logger.logtxt(f'{self.split: >5}: Used shm for {index}', prnt=False)
                except FileNotFoundError:  # Shared memory (cached imagenet) not found, load from disk
                    sample = self.loader(path)
                    # self.logger.logtxt(f'{self.split: >5}: Disk load for {index}', prnt=False)

        except UnidentifiedImageError as e:
            msg = 'ImageNet22k could not load picture at {}. Unidentified image error.'.format(path)
            self.logwarning(msg, e)
            return self[(index + 1) % len(self)]
        except OSError as e:
            msg = 'ImageNet22k could not load picture at {}. OS Error.'.format(path)
            self.logwarning(msg, e)
            return self[(index + 1) % len(self)]
        except sre_constants_error as e:
            msg = 'ImageNet22k could not load picture at {}. SRE Constants Error.'.format(path)
            self.logwarning(msg, e)
            return self[(index + 1) % len(self)]

        if self.transform is not None:
            if self.conditional_transform is not None:
                sample = self.pre_transform(sample)
                sample = self.conditional_transform(sample, target)
                sample = self.post_transform(sample)
            else:
                sample = self.transform(sample)

        return sample, target, index

    def cache(self, ids: List[int]):
        self.cached_images = {}
        mem = 0
        procbar = tqdm(ids, desc=f'Caching {len(ids)} resized images for ImageNet22k (current cache size is {mem: >9.4f} GB)')
        for index in procbar:
            path, target = self.samples[index]
            try:
                sample = self.loader(path)
            except UnidentifiedImageError as e:
                continue
            except OSError as e:
                continue
            except sre_constants_error as e:
                continue
            if isinstance(self.pre_transform.transforms[0], transforms.Resize):
                sample = self.pre_transform.transforms[0](sample)
            elif isinstance(self.transform.transforms[0], transforms.Resize):
                sample = self.transform.transforms[0](sample)
            self.cached_images[index] = sample
            mem += np.prod(sample.size) * 3 * 1e-9
            procbar.set_description(f'Caching {len(ids)} resized images for ImageNet22k (current cache size is {mem: >9.4f} GB)')

    def load_cached(self, id: int) -> PIL.Image.Image:
        if id in self.cached_images:
            return self.cached_images[id]
        else:
            return None

    def _load_image_from_shared_memory(self, index: int) -> PIL.Image.Image:
        # see :method:`MyImageNet._load_image_from_shared_memory` for some documentation on this!
        shm = shared_memory.SharedMemory(name=f'{"imagenet21k" if not self.subset else "imagenet21ksubset"}_train_{index}')
        img = decode_shape_and_image(np.copy(np.ndarray(shm.size, dtype=np.uint8, buffer=shm.buf)))
        shm.close()
        unregister(shm._name, 'shared_memory')
        img = to_pil_image(img)
        return img

    def logwarning(self, s, err):
        if self.logger is not None:
            self.logger.warning(s, print=False)
        else:
            raise err


class ADImageNet21kSubSet(ADImageNet21k):
    """
    This uses the :class:`ADImageNet21k` implementation but looks at a different root folder.
    That is, instead of `root`/imagenet22k/fall11_whole_extracted/ it uses the data found in `root`/imagenet21k_subset/.
    """
    base_folder = 'imagenet21k_subset'
