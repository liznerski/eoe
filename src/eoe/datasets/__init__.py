from copy import deepcopy
from typing import List, Union, Tuple

import numpy as np
from torchvision.transforms import Resize, Compose

from eoe.datasets.bases import TorchvisionDataset, CombinedDataset
from eoe.datasets.cifar import ADCIFAR10, ADCIFAR100
from eoe.datasets.cub import ADCUB
from eoe.datasets.dtd import ADDTD
from eoe.datasets.fmnist import ADFMNIST
from eoe.datasets.imagenet import ADImageNet, ADImageNet21k, ADImageNet21kSubSet
from eoe.datasets.imagenetoe import ADImageNetOE
from eoe.datasets.mnist import ADMNIST, ADEMNIST
from eoe.datasets.mvtec import ADMvTec
from eoe.datasets.tinyimages import ADTinyImages
from eoe.datasets.custom import ADCustomDS
from eoe.utils.logger import Logger
from eoe.utils.transformations import TRANSFORMS, get_transform, ConditionalCompose

DS_CHOICES = {  # list of implemented datasets (most can also be used as OE)
    'cifar10': {
        'class': ADCIFAR10, 'default_size': 32, 'no_classes': 10, 'oe_only': False,
        'str_labels':  ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    },
    'imagenet': {
        'class': ADImageNet, 'default_size': 256, 'no_classes': 30, 'oe_only': False,
        'str_labels': deepcopy(ADImageNet.ad_classes),
    },
    'cifar100': {
        'class': ADCIFAR100, 'default_size': 32, 'no_classes': 100, 'oe_only': False,
        'str_labels': [
            'beaver', 'dolphin', 'otter', 'seal', 'whale',
            'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
            'orchid', 'poppy', 'rose', 'sunflower', 'tulip',
            'bottle', 'bowl', 'can', 'cup', 'plate',
            'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper',
            'clock', 'keyboard', 'lamp', 'telephone', 'television',
            'bed', 'chair', 'couch', 'table', 'wardrobe',
            'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
            'bear', 'leopard', 'lion', 'tiger', 'wolf',
            'bridge', 'castle', 'house', 'road', 'skyscraper',
            'cloud', 'forest', 'mountain', 'plain', 'sea',
            'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
            'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
            'crab', 'lobster', 'snail', 'spider', 'worm',
            'baby', 'boy', 'girl', 'man', 'woman',
            'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
            'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
            'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree',
            'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train',
            'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'
        ]
    },
    'imagenet21k': {
        'class': ADImageNet21k, 'default_size': 256, 'no_classes': 21811, 'oe_only': False,
        'str_labels': [str(i) for i in range(21811)],  # ?
    },
    'imagenet21ksubset': {
        'class': ADImageNet21kSubSet, 'default_size': 256, 'no_classes': 21811, 'oe_only': False,
        'str_labels': [str(i) for i in range(21811)],  # ?
    },
    'tinyimages': {
        'class': ADTinyImages, 'default_size': 32, 'no_classes': 1, 'oe_only': False, 'str_labels': ['tiny_image'],
    },
    'mvtec': {
        'class': ADMvTec, 'default_size': 256, 'no_classes': 15, 'oe_only': False,
        'str_labels': [
            'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
            'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
            'wood', 'zipper'
        ]
    },
    'imagenetoe': {
        'class': ADImageNetOE, 'default_size': 256, 'no_classes': 1000, 'oe_only': True,
        'str_labels':  list(range(1000)),  # not required
    },
    'fmnist': {
        'class': ADFMNIST, 'default_size': 28, 'no_classes': 10, 'oe_only': False,
        'str_labels': [
            'top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boot'
        ]
    },
    'cub': {
        'class': ADCUB, 'default_size': 256, 'no_classes': 200, 'oe_only': False,
        'str_labels': [
            'Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani', 'Crested_Auklet',
            'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird', 'Red_winged_Blackbird',
            'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting', 'Lazuli_Bunting', 'Painted_Bunting',
            'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat', 'Eastern_Towhee', 'Chuck_will_Widow',
            'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant', 'Bronzed_Cowbird', 'Shiny_Cowbird', 'Brown_Creeper',
            'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo', 'Mangrove_Cuckoo', 'Yellow_billed_Cuckoo',
            'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker', 'Acadian_Flycatcher', 'Great_Crested_Flycatcher',
            'Least_Flycatcher', 'Olive_sided_Flycatcher', 'Scissor_tailed_Flycatcher', 'Vermilion_Flycatcher',
            'Yellow_bellied_Flycatcher', 'Frigatebird', 'Northern_Fulmar', 'Gadwall', 'American_Goldfinch',
            'European_Goldfinch', 'Boat_tailed_Grackle', 'Eared_Grebe', 'Horned_Grebe', 'Pied_billed_Grebe', 'Western_Grebe',
            'Blue_Grosbeak', 'Evening_Grosbeak', 'Pine_Grosbeak', 'Rose_breasted_Grosbeak', 'Pigeon_Guillemot',
            'California_Gull', 'Glaucous_winged_Gull', 'Heermann_Gull', 'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull',
            'Slaty_backed_Gull', 'Western_Gull', 'Anna_Hummingbird', 'Ruby_throated_Hummingbird', 'Rufous_Hummingbird',
            'Green_Violetear', 'Long_tailed_Jaeger', 'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay',
            'Dark_eyed_Junco', 'Tropical_Kingbird', 'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher', 'Pied_Kingfisher',
            'Ringed_Kingfisher', 'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon', 'Mallard',
            'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird', 'Nighthawk', 'Clark_Nutcracker',
            'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole', 'Orchard_Oriole', 'Scott_Oriole', 'Ovenbird',
            'Brown_Pelican', 'White_Pelican', 'Western_Wood_Pewee', 'Sayornis', 'American_Pipit', 'Whip_poor_Will',
            'Horned_Puffin', 'Common_Raven', 'White_necked_Raven', 'American_Redstart', 'Geococcyx', 'Loggerhead_Shrike',
            'Great_Grey_Shrike', 'Baird_Sparrow', 'Black_throated_Sparrow', 'Brewer_Sparrow', 'Chipping_Sparrow',
            'Clay_colored_Sparrow', 'House_Sparrow', 'Field_Sparrow', 'Fox_Sparrow', 'Grasshopper_Sparrow', 'Harris_Sparrow',
            'Henslow_Sparrow', 'Le_Conte_Sparrow', 'Lincoln_Sparrow', 'Nelson_Sharp_tailed_Sparrow', 'Savannah_Sparrow',
            'Seaside_Sparrow', 'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow', 'White_crowned_Sparrow',
            'White_throated_Sparrow', 'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow', 'Cliff_Swallow',
            'Tree_Swallow', 'Scarlet_Tanager', 'Summer_Tanager', 'Artic_Tern', 'Black_Tern', 'Caspian_Tern',
            'Common_Tern', 'Elegant_Tern', 'Forsters_Tern', 'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher',
            'Sage_Thrasher', 'Black_capped_Vireo', 'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo',
            'Warbling_Vireo', 'White_eyed_Vireo', 'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 'Black_and_white_Warbler',
            'Black_throated_Blue_Warbler', 'Blue_winged_Warbler', 'Canada_Warbler', 'Cape_May_Warbler', 'Cerulean_Warbler',
            'Chestnut_sided_Warbler', 'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler',
            'Mourning_Warbler', 'Myrtle_Warbler', 'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler',
            'Prairie_Warbler', 'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler',
            'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush', 'Bohemian_Waxwing',
            'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker', 'Red_bellied_Woodpecker',
            'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker', 'Bewick_Wren', 'Cactus_Wren',
            'Carolina_Wren', 'House_Wren', 'Marsh_Wren', 'Rock_Wren', 'Winter_Wren', 'Common_Yellowthroat'
        ]
    },
    'dtd': {
        'class': ADDTD, 'default_size': 256, 'no_classes': 47, 'oe_only': False,
        'str_labels': [
            'banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered', 'cobwebbed', 'cracked', 'crosshatched',
            'crystalline', 'dotted', 'fibrous', 'flecked', 'freckled', 'frilly', 'gauzy', 'grid', 'grooved', 'honeycombed',
            'interlaced', 'knitted', 'lacelike', 'lined', 'marbled', 'matted', 'meshed', 'paisley', 'perforated', 'pitted',
            'pleated', 'polka-dotted', 'porous', 'potholed', 'scaly', 'smeared', 'spiralled', 'sprinkled', 'stained',
            'stratified', 'striped', 'studded', 'swirly', 'veined', 'waffled', 'woven', 'wrinkled', 'zigzagged'
        ]
    },
    'mnist': {
        'class': ADMNIST, 'default_size': 28, 'no_classes': 10, 'oe_only': False,
        'str_labels': [
          "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"
        ]
    },
    'emnist': {
        'class': ADEMNIST, 'default_size': 28, 'no_classes': 26, 'oe_only': False, 'str_labels': list(range(26)),  # ?
    },
}

TRAIN_NOMINAL_ID = 0
TRAIN_OE_ID = 1
TEST_NOMINAL_ID = 2
TEST_ANOMALOUS_ID = 3
DS_PARTS = {  # all possible dataset parts (see MSM)
    'train_nominal': TRAIN_NOMINAL_ID, 'train_oe': TRAIN_OE_ID,
    'test_nominal': TEST_NOMINAL_ID, 'test_anomalous': TEST_ANOMALOUS_ID
}


class MSM(object):
    def __init__(self, transform: str, ds_part: str, magnitude: int = None):
        """
        An MSM (multi-scale mode) consists of a transformation and a string defining a `type` of data.
        `Type` here refers to one of the four DS_PARTS, which are
            - normal training samples (`train_nominal`)
            - outlier exposure samples (`train_oe`)
            - normal test samples (`test_nominal`)
            - anomalous test samples (`test_anomalous`)
        MSMs can be passed to :method:`load_dataset`, which parses them to create conditional transforms for the datasets.
        For instance, the MSM with the transform `lpf` and the type `train_nominal` will apply a low pass filter to
        normal training samples only.
        Note that this is a rather atypical kind of data augmentation since it uses labels, which, during testing,
        is typically illegitimate. We used it to experiment with different version of frequency filters in our
        frequency analysis experiments, however, have only reported results for equal filters on all data parts in the paper.
        @param transform: a string defining the transform;
            all implemented MSM transforms are defined in :attr:`eoe.utils.transsformations.TRANSFORMS`.
        @param ds_part: a string defining the `type` (dataset part);
            all available types are defined in :attr:`DS_PARTS`.
        @param magnitude: the magnitude of the transformation. The meaning of magnitude may vary depending on the transformation.
            For a description of the magnitude have a look at Appendix C in our paper or `eoe.utils.transformations`.
            In general, the larger the magnitude the more severe the transformation. This argument is optional because
            the magnitude can also be set later via :method:`set_magnitude`. However, the MSM cannot be applied until
            a magnitude is set.
        """
        assert transform in TRANSFORMS
        assert ds_part in DS_PARTS
        self.transform = TRANSFORMS[transform]
        self.ds_part = DS_PARTS[ds_part]
        self.transform_str = transform
        self.ds_part_str = ds_part
        self.magnitude = magnitude

    def set_magnitude(self, magnitude: int) -> 'MSM':
        """ sets the magnitude of the transformation, which is used by the transformations  """
        self.magnitude = magnitude
        return self

    def get_transform(self):
        """ returns the transformation function defined by :attr:`transform_str` with the current magnitude """
        return get_transform(self.transform, self.magnitude)

    def __str__(self):
        return '+'.join((self.transform_str, self.ds_part_str)) + f"--M{self.magnitude}"

    def __repr__(self):
        return str(self)

    @staticmethod
    def load(msm: str, load_magnitude=True) -> 'MSM':
        """
        Splits the string `TRANSFORM+DSPART--M` into TRANSFORM that defines the transform, DSPART that defines the `type`,
        and M that defines the magnitude. `--M` is optional. Creates an MSM object with the retrieved parameters.
        @param msm: the combined string representation that is to be split.
        @param load_magnitude: whether to actually load the magnitude from `--M` or ignore it.
        @return: an MSM with the retrieved transform, type, and magnitude.
        """
        transform, ds_part = msm.split('+')
        magnitude = None
        if '--M' in ds_part:  # backward compatibility (old version don't have --M)
            ds_part, magnitude = ds_part.split('--M')
        res = MSM(transform, ds_part)
        if load_magnitude and magnitude is not None:
            res.set_magnitude(int(magnitude))
        return res


def get_raw_shape(train_transform: Compose, dataset_name: str) -> Tuple[int, int, int]:
    """ detects the raw_shape of the data (i.e., the shape before clipping etc.) using the first resize transform """
    if len(train_transform.transforms) > 0 and isinstance(train_transform.transforms[0], Resize):
        t = train_transform.transforms[0]
        if isinstance(t.size, int):
            return (3, t.size, t.size)
        else:
            return (3, *t.size)
    else:
        size = DS_CHOICES[dataset_name]['default_size']
        return 3, size, size


def load_dataset(dataset_name: str, data_path: str, normal_classes: List[int], nominal_label: int,
                 train_transform: Compose, test_transform: Compose, logger: Logger = None,
                 oe_name: str = None, oe_limit_samples: Union[int, List[int]] = np.infty, oe_limit_classes: int = np.infty,
                 msms: List[MSM] = ()) -> TorchvisionDataset:
    """
    Prepares a dataset, includes setting up all the necessary attributes such as a list of filepaths and labels.
    Requires a list of normal classes that determines the labels and which classes are available during training.
    If an OE dataset is specified, prepares a combined dataset.
    The combined dataset's test split is the test split of the normal dataset.
    The combined dataset's training split is a combination of the normal training data and the OE data.
    It provides a balanced concatenated data loader. See :class:`eoe.datasets.bases.CombinedDataset`.

    @param dataset_name: Defines the normal dataset (containing also anomalous test samples). See :attr:`DS_CHOICES`.
    @param data_path: Defines the root directory for all datasets. Most of them get automatically downloaded if not present
        at this directory. Each dataset has its own subdirectory (e.g., eoe/data/datasets/imagenet/).
    @param normal_classes: A list of normal classes. Normal training samples are all from these classes.
        Samples from other classes are not available during training. During testing, other classes will have anomalous labels.
    @param nominal_label: The integer defining the normal (==nominal) label. Usually 0.
    @param train_transform: preprocessing pipeline used for training, includes all kinds of image transformations.
        May contain the dummy transformation 'norm' that will be replaced with a torchvision normalization instance.
        The required mean and std of the normal training data will be extracted automatically.
    @param test_transform: preprocessing pipeline used for testing,
        includes all kinds of image transformations but no data augmentation.
        May contain the dummy transformation 'norm' that will be replaced with a torchvision normalization instance.
        The required mean and std of the normal training data will be extracted automatically.
    @param logger: Optional. Some logger instance. Is only required for logging warnings related to the datasets.
    @param oe_name: Optional. Defines the OE dataset. See method description.
    @param oe_limit_samples: Optional. If given, limits the number of different OE samples. That is, instead of using the
        complete OE dataset, creates a subset to be used as OE. If `oe_limit_samples` is an integer, samples a random subset
        with the provided size. If `oe_limit_samples` is a list of integers, create a subset with the indices provided.
    @param oe_limit_classes: Optional. If given, limits the number of different classes of OE samples. That is, instead of
        using the complete OE dataset, creates a subset consisting only of OE images from randomly selected classes.
        Note that the typical OE dataset implementations (80MTI, ImageNet-21k) come without classes. For these OE datasets
        this parameter has no effect.
    @param msms: A list of MSMs that are to be applied to the dataset samples. See :class:`MSM` above.
    @return: the prepared TorchvisionDataset instance.
    """

    assert dataset_name in DS_CHOICES, f'{dataset_name} is not in {DS_CHOICES}'

    raw_shape = get_raw_shape(train_transform, dataset_name)

    def get_ds(name: str, normal_dataset: TorchvisionDataset = None):
        if normal_dataset is None:
            train_classes = normal_classes
            train_label = nominal_label
            total_train_transform = train_transform
            total_test_transform = test_transform
            train_conditional_transform = ConditionalCompose([
                (nominal_label, msm.get_transform(), None) for msm in msms if msm.ds_part == TRAIN_NOMINAL_ID
            ])
            test_conditional_transform = ConditionalCompose([
                (nominal_label,
                 msm.get_transform() if msm.ds_part == TEST_NOMINAL_ID else None,
                 msm.get_transform() if msm.ds_part == TEST_ANOMALOUS_ID else None)
                for msm in msms if msm.ds_part in (TEST_NOMINAL_ID, TEST_ANOMALOUS_ID)
            ])
            limit = np.infty
            kwargs = {}
        else:  # oe case
            train_classes = sorted(
                np.random.choice(list(range(no_classes(name))), min(no_classes(name), oe_limit_classes), False)
            )
            train_label = 1 - nominal_label
            total_train_transform = deepcopy(normal_dataset.train_transform)
            total_test_transform = deepcopy(normal_dataset.test_transform)
            limit = oe_limit_samples
            train_conditional_transform = ConditionalCompose([
                (nominal_label, msm.get_transform(), msm.get_transform()) for msm in msms if msm.ds_part == TRAIN_OE_ID
            ])
            test_conditional_transform = None
            kwargs = {}
            if isinstance(normal_dataset, ADCustomDS) and name == 'custom':  # special case for custom being used as OE
                if oe_limit_classes < np.inf:
                    raise ValueError(
                        "Using the custom dataset with its own OE part cannot be combined with limiting the OE classes."
                    )
                train_classes = normal_classes
                train_label = nominal_label
                kwargs = {"oe": True}
        args = (
            data_path, train_classes, train_label, total_train_transform, total_test_transform, raw_shape, logger, limit,
            train_conditional_transform, test_conditional_transform
        )

        if DS_CHOICES[name]['oe_only']:
            assert normal_dataset is not None, f"{name} can only be used as OE!"
            dataset = DS_CHOICES[name]['class'](*args, **kwargs)
        else:
            dataset = DS_CHOICES[name]['class'](*args, **kwargs)

        if normal_dataset is not None:  # oe case
            dataset.gpu_train_transform = Compose([normal_dataset.gpu_train_transform, dataset.gpu_train_transform])
            dataset.gpu_test_transform = Compose([normal_dataset.gpu_test_transform, dataset.gpu_test_transform])
        return dataset

    dataset = get_ds(dataset_name)
    if oe_name is not None:
        oe = get_ds(oe_name, dataset)
        dataset = CombinedDataset(dataset, oe)

    return dataset


def no_classes(dataset_name: str) -> int:
    """ returns the number of classes for the given dataset """
    return DS_CHOICES[dataset_name]['no_classes']


def str_labels(dataset_name) -> List[str]:
    """ returns a list of class descriptions for the given dataset """
    return DS_CHOICES[dataset_name]['str_labels']

