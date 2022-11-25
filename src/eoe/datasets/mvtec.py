import os
import os.path as pt
import shutil
import tarfile
from typing import List, Tuple, Union
from collections import Counter

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Subset
from torchvision.datasets import VisionDataset
from torchvision.datasets.imagenet import check_integrity, verify_str_arg
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from eoe.datasets.bases import TorchvisionDataset
from eoe.utils.logger import Logger
from eoe.utils.transformations import ConditionalCompose


class ADMvTec(TorchvisionDataset):
    def __init__(self, root: str, normal_classes: List[int], nominal_label: int,
                 train_transform: transforms.Compose, test_transform: transforms.Compose,
                 raw_shape: Tuple[int, int, int], logger: Logger = None, limit_samples: Union[int, List[int]] = np.infty,
                 train_conditional_transform: ConditionalCompose = None, test_conditional_transform: ConditionalCompose = None):
        """
        AD dataset for MVTec-AD (https://www.mvtec.com/company/research/datasets/mvtec-ad).
        If no MVTec data is found in the root directory, the data is downloaded and processed to be stored in
        torch tensors with appropriate size (defined by raw_shape).
        This speeds up data loading at the start of training.
        Implements :class:`eoe.datasets.bases.TorchvisionDataset`.
        """
        super().__init__(
            root, normal_classes, nominal_label, train_transform, test_transform, 15, raw_shape, logger, limit_samples,
            train_conditional_transform, test_conditional_transform
        )

        self._train_set = MvTec(
            self.root, self.raw_shape, 'train', self.train_transform, self.target_transform,
            logger=logger, conditional_transform=self.train_conditional_transform
        )
        self._train_set = self.create_subset(self._train_set, self._train_set.targets)

        # MvTec.targets refers to the class label ('bottle', etc.),
        # but for the actual test label we need to use the defect label that, per class, marks a sample as healthy or defect.
        # Note that the target_transform for training still is the default target_transform that maps the class label
        # to the nominal label. The training set contains only healthy instances anyway.
        self.defect_label_transform = transforms.Compose([transforms.Lambda(
            lambda x: self.anomalous_label if x != MvTec.normal_defect_label_idx else self.nominal_label
        )])
        self._test_set = MvTec(
            self.root, self.raw_shape, 'test_defect_label_target', self.test_transform,
            self.defect_label_transform, logger=logger, conditional_transform=self.test_conditional_transform, enlarge=False
        )
        # MVTec-AD doesn't use the one vs. rest AD benchmark but instead comes with a set of ground-truth anomalies (defects)
        # per (normal) class. Thus, we have to exclude samples from other classes for the test set.
        self._test_set = Subset(
            self._test_set,
            np.argwhere(
                np.isin(np.asarray(self._test_set.targets), self.normal_classes)
            ).flatten().tolist()
        )

    def _get_raw_train_set(self):
        train_set = MvTec(
            self.root, self.raw_shape, 'train',
            transforms.Compose([transforms.Resize((self.raw_shape[-1])), transforms.ToTensor(), ]), self.target_transform,
            download=True, enlarge=False,
        )
        return Subset(
            train_set,
            np.argwhere(
                np.isin(np.asarray(train_set.targets), self.normal_classes)
            ).flatten().tolist()
        )

    def n_normal_anomalous(self, train=True) -> dict:
        """
        Extract the number of normal and anomalous data samples. Overwrites base because :attr:`MVTec.targets` refers, as usual,
        to the class labels. The test split of MVTec looks into :attr:`MVTec.anomaly_labels` instead, which per class defines
        whether the sample is a healthy/normal sample or defected/anomalous sample. This is why we need to use
        a different target_transform than the default one in self.target_transform.
        @param train: Whether to consider training or test samples.
        @return: A dictionary like {0: #normal_samples, 1: #anomalous_samples} (may change depending on the nominal label)
        """
        if train:
            ds = self.train_set
            return dict(Counter([self.target_transform(t) for t in np.asarray(ds.dataset.targets)[ds.indices]]))
        else:
            ds = self.test_set
            return dict(Counter([self.defect_label_transform(t) for t in np.asarray(ds.dataset.anomaly_labels)[ds.indices]]))


class MvTec(VisionDataset):
    url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
    base_folder = 'mvtec'  # appended to root directory as a subdirectory
    labels = (
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper'
    )  # all classes
    normal_defect_label = 'good'  # the subfolder of each class that contains the normal samples
    normal_defect_label_idx = 0  # the id used for marking normal/healthy samples

    def __init__(self, root: str, shape: Tuple[int, int, int], split: str,
                 transform: transforms.Compose = None, target_transform: transforms.Compose = None,
                 download=True, logger: Logger = None, conditional_transform: ConditionalCompose = None, enlarge=True):
        """
        Implements a torchvision-style vision dataset.
        Loads the images from the disk and prepares them in torch tensors of correct shape, which are stored on the disk.
        Later instances of this dataset with the same shape will load the torch tensors instead, which is faster.
        Also, automatically loads the raw data from the web if not found in `root`/mvtec/.

        @param root: Directory where the data is found or downloaded to.
        @param shape: The shape (c x h x w) of the prepared torch tensors containing the images.
        @param split: whether to use "train", "test", or "test_defect_label_target" data.
            In the latter case the get_item method returns labels indexing the anomalous class rather than
            the object class. That is, instead of returning 0 for "bottle", it returns "1" for "large_broken".
        @param transform: A preprocessing pipeline of image transformations.
        @param target_transform: A preprocessing pipeline of label (integer) transformations.
        @param download: Whether to download the data if not found in `root`/mvtec/.
        @param logger: Optional. Some logger instance. Only used to log warnings.
        @param conditional_transform: Optional. A preprocessing pipeline of conditional image transformations.
            See :class:`eoe.datasets.bases.TorchvisionDataset`. Usually this is None.
        @param enlarge: Whether to enlarge the dataset by repeating all training samples ten times.
            This slightly speed up data loading.
        """
        super(MvTec, self).__init__(root, transform=transform, target_transform=target_transform)
        self.shape = shape
        self.transforms = transform
        self.logger = logger
        self.enlarge = enlarge
        self.conditional_transform = conditional_transform
        self.pre_transform, self.post_transform = None, None
        if self.transform is not None and self.conditional_transform is not None:
            # splits transform at ToTensor(); apply pre_transform - conditional_transform - post_transform (including ToTensor())
            totensor_pos = [isinstance(t, transforms.ToTensor) for t in self.transform.transforms]
            totensor_pos = totensor_pos.index(True) if True in totensor_pos else 0
            self.pre_transform = transforms.Compose(self.transform.transforms[:totensor_pos])
            self.post_transform = transforms.Compose(self.transform.transforms[totensor_pos:])

        self.split = verify_str_arg(split, "split", ("train", "test", "test_defect_label_target"))

        if download:
            self.download(self.shape[1:])

        print('Loading dataset from {}...'.format(self.data_file))
        dataset_dict = torch.load(self.data_file)
        self.anomaly_label_strings = dataset_dict['anomaly_label_strings']
        if self.split == 'train':
            self.data, self.targets = dataset_dict['train_data'], dataset_dict['train_labels']
            self.anomaly_labels = None
        else:
            self.data, self.targets = dataset_dict['test_data'], dataset_dict['test_labels']
            self.anomaly_labels = dataset_dict['test_anomaly_labels']

        if self.enlarge:
            self.data, self.targets = self.data.repeat(10, 1, 1, 1), self.targets.repeat(10)
            self.anomaly_labels = self.anomaly_labels.repeat(10) if self.anomaly_labels is not None else None

    @property
    def data_file(self):
        return os.path.join(self.root, self.base_folder, self.filename)

    @property
    def filename(self):
        return "admvtec_{}x{}.pt".format(self.shape[1], self.shape[2])

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int, int], Tuple[torch.Tensor, int, torch.Tensor, int]]:
        img, label = self.data[index], self.targets[index]

        if self.split == 'test_defect_label_target':
            label = self.target_transform(self.anomaly_labels[index])
        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.transform is not None:
            img = to_pil_image(img)
            if self.conditional_transform is not None:
                img = self.pre_transform(img)
                img = self.conditional_transform(img, label)
                img = self.post_transform(img)
            else:
                img = self.transform(img)

        return img, label, index

    def __len__(self) -> int:
        return len(self.data)

    def download(self, shape, verbose=True):
        if not check_integrity(self.data_file):
            tmp_dir = pt.join(self.root, self.base_folder, 'tmp')
            os.makedirs(tmp_dir, exist_ok=False)
            self.download_and_extract_archive(
                self.url, os.path.join(self.root, self.base_folder), extract_root=tmp_dir,
            )
            train_data, train_labels = [], []
            test_data, test_labels, test_maps, test_anomaly_labels = [], [], [], []
            anomaly_labels, albl_idmap = [], {self.normal_defect_label: self.normal_defect_label_idx}

            for lbl_idx, lbl in enumerate(self.labels):
                if verbose:
                    print('Processing data for label {}...'.format(lbl))
                for anomaly_label in sorted(os.listdir(os.path.join(tmp_dir, lbl, 'test'))):
                    for img_name in sorted(os.listdir(os.path.join(tmp_dir, lbl, 'test', anomaly_label))):
                        with open(os.path.join(tmp_dir, lbl, 'test', anomaly_label, img_name), 'rb') as f:
                            sample = Image.open(f)
                            sample = self.img_to_torch(sample, shape)
                        if anomaly_label != self.normal_defect_label:
                            mask_name = self.convert_img_name_to_mask_name(img_name)
                            with open(os.path.join(tmp_dir, lbl, 'ground_truth', anomaly_label, mask_name), 'rb') as f:
                                mask = Image.open(f)
                                mask = self.img_to_torch(mask, shape)
                        else:
                            mask = torch.zeros_like(sample)
                        test_data.append(sample)
                        test_labels.append(lbl_idx)
                        test_maps.append(mask)
                        if anomaly_label not in albl_idmap:
                            albl_idmap[anomaly_label] = len(albl_idmap)
                        test_anomaly_labels.append(albl_idmap[anomaly_label])

                for anomaly_label in sorted(os.listdir(os.path.join(tmp_dir, lbl, 'train'))):
                    for img_name in sorted(os.listdir(os.path.join(tmp_dir, lbl, 'train', anomaly_label))):
                        with open(os.path.join(tmp_dir, lbl, 'train', anomaly_label, img_name), 'rb') as f:
                            sample = Image.open(f)
                            sample = self.img_to_torch(sample, shape)
                        train_data.append(sample)
                        train_labels.append(lbl_idx)

            anomaly_labels = list(zip(*sorted(albl_idmap.items(), key=lambda kv: kv[1])))[0]
            train_data = torch.stack(train_data)
            train_labels = torch.IntTensor(train_labels)
            test_data = torch.stack(test_data)
            test_labels = torch.IntTensor(test_labels)
            test_maps = torch.stack(test_maps)[:, 0, :, :]  # r=g=b -> grayscale
            test_anomaly_labels = torch.IntTensor(test_anomaly_labels)
            torch.save({
                    'train_data': train_data, 'train_labels': train_labels, 'test_data': test_data,
                    'test_labels': test_labels, 'test_maps': test_maps,
                    'test_anomaly_labels': test_anomaly_labels, 'anomaly_label_strings': anomaly_labels
                }, self.data_file
            )

            # cleanup temp directory
            try:
                shutil.rmtree(tmp_dir)
            except PermissionError:
                print(f'WARNING: temporary directory at {tmp_dir} could not be removed due to missing permission.')
        else:
            print('Files already downloaded.')
            return

    @staticmethod
    def img_to_torch(img, shape=None):
        if shape is not None:
            return torch.nn.functional.interpolate(
                torch.from_numpy(np.array(img.convert('RGB'))).float().transpose(0, 2).transpose(1, 2)[None, :],
                shape
            )[0].byte()
        else:
            return torch.from_numpy(
                np.array(img.convert('RGB'))
            ).float().transpose(0, 2).transpose(1, 2)[None, :][0].byte()

    @staticmethod
    def convert_img_name_to_mask_name(img_name):
        return img_name.replace('.png', '_mask.png')

    @staticmethod
    def download_and_extract_archive(url, download_root, extract_root=None, filename=None, remove_finished=False):
        download_root = os.path.expanduser(download_root)
        if extract_root is None:
            extract_root = download_root
        if not filename:
            filename = os.path.basename(url)
        if not os.path.exists(download_root):
            os.makedirs(download_root)

        MvTec.download_url(url, download_root, filename)

        archive = os.path.join(download_root, filename)
        print("Extracting {} to {}".format(archive, extract_root))
        MvTec.extract_archive(archive, extract_root, remove_finished)

    @staticmethod
    def download_url(url, root, filename=None):
        """Download a file from a url and place it in root.
        Args:
            url (str): URL to download file from
            root (str): Directory to place downloaded file in
            filename (str, optional): Name to save the file under. If None, use the basename of the URL
        """
        from six.moves import urllib

        root = os.path.expanduser(root)
        if not filename:
            filename = os.path.basename(url)
        fpath = os.path.join(root, filename)

        os.makedirs(root, exist_ok=True)

        def gen_bar_updater():
            pbar = tqdm(total=None)

            def bar_update(count, block_size, total_size):
                if pbar.total is None and total_size:
                    pbar.total = total_size
                progress_bytes = count * block_size
                pbar.update(progress_bytes - pbar.n)

            return bar_update

        # check if file is already present locally
        if not check_integrity(fpath, None):
            try:
                print('Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())
            except (urllib.error.URLError, IOError) as e:
                if url[:5] == 'https':
                    url = url.replace('https:', 'http:')
                    print('Failed download. Trying https -> http instead.'
                          ' Downloading ' + url + ' to ' + fpath)
                    urllib.request.urlretrieve(
                        url, fpath,
                        reporthook=gen_bar_updater()
                    )
                else:
                    raise e
            # check integrity of downloaded file
            if not check_integrity(fpath, None):
                raise RuntimeError("File not found or corrupted.")

    @staticmethod
    def extract_archive(from_path, to_path=None, remove_finished=False):
        if to_path is None:
            to_path = os.path.dirname(from_path)

        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)
