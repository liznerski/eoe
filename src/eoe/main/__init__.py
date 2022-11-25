import json
import os
import os.path as pt
import time
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from itertools import product
from typing import Callable, Tuple
from typing import List
from typing import Union
from warnings import filterwarnings

import cv2
import deap.base
import numpy as np
import torch
from deap import base, creator, tools
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose

from eoe.datasets import DS_CHOICES as IMG_DS_CHOICES, load_dataset, DS_PARTS
from eoe.datasets import MSM
from eoe.datasets import TRAIN_NOMINAL_ID, TRAIN_OE_ID
from eoe.datasets import no_classes
from eoe.evolve import mutate_individual, mate_individuals, init_individual, evaluate_individual, select_individual
from eoe.evolve import replace_individuals_randomly
from eoe.evolve.tree import EvolNode, Tree, Individual
from eoe.training import TRAINER
from eoe.training.ad_trainer import ADTrainer
from eoe.utils.logger import Logger
from eoe.utils.logger import SetupEncoder
from eoe.utils.logger import time_format
from eoe.utils.transformations import TRANSFORMS

filterwarnings(action='ignore', category=DeprecationWarning, module='torch')
cv2.setNumThreads(0)  # possible deadlock fix?


def default_argsparse(modify_descr: Callable[[str, ], str], modify_parser: Callable[[ArgumentParser], None] = None,
                      modify_args: Callable[[Namespace], None] = None) -> Namespace:
    """
    Creates and applies the argument parser for all training scripts.
    @param modify_descr: function that modifies the default description.
    @param modify_parser: function that modifies the default parser provided by this method.
        Can be used to, e.g., add further arguments or change the default values for arguments.
    @param modify_args: function that modifies the actual arguments retrieved by the parser.
    @return: the parsed arguments.
    """
    parser = ArgumentParser(
        description=modify_descr(
            "Iterates over a set of classes found in the dataset and multiple random seeds per class. "
            "For each class-seed combination, it trains and evaluates a given AD model and objective. "
            "Depending on the ad_mode, it either treats the current class or all but the current class as normal. "
            "It always evaluates using the full test set. "
        )
    )
    parser.add_argument(
        '-ds', '--dataset', type=str, default=None, choices=IMG_DS_CHOICES,
        help="The dataset for which to train the AD model. All datasets have an established train and test split. "
             "During training, use only normal samples from this dataset. For testing, use all samples. "
    )
    parser.add_argument(
        '-oe', '--oe-dataset', type=str, default=None, choices=IMG_DS_CHOICES + ('none', ),
        help="Optional Outlier Exposure (OE) dataset. If given, concatenate an equally sized batch of random "
             "samples from this dataset to the batch of normal training samples from the main dataset. "
             "These concatenated samples are used as auxiliary anomalies. "
    )
    parser.add_argument(
        '--oe-size', type=int, default=np.infty,
        help="Optional. If given, uses a random subset of the OE dataset as OE with the subset having the provided size. "
    )
    parser.add_argument(
        '-b', '--batch-size', type=int, default=200,
        help="The batch size. If there is an OE dataset, the overall batch size will be twice as large as an equally "
             "sized batch of OE samples gets concatenated to the batch of normal training samples."
    )
    parser.add_argument(
        '-e', '--epochs', type=int, default=50,
        help="How many full iterations of the dataset are to be trained per class-seed combination."
    )
    parser.add_argument(
        '-lr', '--learning-rate', type=float, default=1e-3,
        help="The initial learning rate."
    )
    parser.add_argument(
        '-wdk', '--weight-decay', type=float, default=1e-4,
        help="The weight decay."
    )
    parser.add_argument(
        '--milestones', type=int, nargs='+', default=[],
        help="Milestones for the learning rate scheduler; at each milestone the learning rate is reduced by 0.1."
    )
    parser.add_argument(
        '-o', '--objective', type=str, default='hsc', choices=TRAINER.keys(),
        help="This defines the objective with which the AD models are trained. It determines both the loss and anomaly score."
             "Some objective may require certain network architectures (e.g., autoencoders). "
    )
    parser.add_argument(
        '--ad-mode', type=str, default='ovr', choices=('ovr', 'loo'),
        help="The anomaly detection (AD) benchmark mode. Supports one vs. rest, where the `current` class is "
             "considered normal and the rest classes anomalous, and leave one class out, where the `current` class is "
             "considered anomalous and the rest classes normal."
    )
    parser.add_argument(
        '--classes', type=int, nargs='+', default=None,
        help='Defines the set of classes that are iterated over. In each iteration the `current` class is '
             'treated as defined by the `--ad-mode`. Defaults to all available classes of the given dataset. '
    )
    parser.add_argument(
        '-d', '--devices', type=int, metavar='GPU-ID', default=None,
        help="Which device to use for training. "
             "Defaults to the first available, which is either the first GPU if a GPU is available or to the CPU otherwise. "
             "CPU training will be very slow. Parallel training on multiple GPUs with one script is not implemented. "
    )
    parser.add_argument(
        '-it', '--iterations', type=int, default=2,
        help="The number of iterations for each class with different random seeds. "
    )
    parser.add_argument(
        '--load', type=str, metavar='FILE-PATH', default=None,
        help="Optional. If provided, needs to be a path to a logging directory of a previous experiment. "
             "Load the configuration from the previous experiment where available. Some configurations need to be matched "
             "manually, such as the data transformation pipelines. "
             "If the same default configuration is used (e.g., `train_cifar.py`), no matching will be required. "
             "Then, load the model snapshots and training state and continue the experiment. "
             "The trainer will start with reevaluating all completed classes and seeds, "
             "which should yield the same metrics again. "
             "For unfinished and not-yet-started class-seed combinations, train and evaluate as usual. "
             "Create a new logging directory by concatenating `---CNTD` to the old directory name. "
    )
    parser.add_argument(
        '--comment', type=str, default='',
        help="Optional. This string will be concatenated to the default logging directory name, which is `log_YYYYMMDDHHMMSS`, "
             "where the latter is the current datetime."
    )
    parser.add_argument(
        '--superdir', type=str, default=".",
        help='Optional. If this run does not continue a previous run, the script will create a new '
             'logging directory `eoe/data/results/log_YYYYMMDDHHMMSS`. `--superdir` will change this to '
             '`eoe/data/results/SUPERDIR/log_log_YYYYMMDDHHMMSS`, where SUPERDIR is the string given via this argument. '
    )
    if modify_parser is not None:
        modify_parser(parser)
    args = parser.parse_args()
    if args.devices is None:
        args.devices = list(range(torch.cuda.device_count()))
    if args.oe_dataset == 'none':
        args.oe_dataset = None
    args.ad_mode = {'ovr': 'one_vs_rest', 'loo': 'leave_one_out', 'ff': 'fifty_fifty'}[args.ad_mode]
    if modify_args is not None:
        modify_args(args)
    return args


def ms_argsparse(modify_descr: Callable[[str, ], str], modify_parser: Callable[[ArgumentParser], None] = None,
                 modify_args: Callable[[Namespace], None] = None):
    """ adds the `--ms-mode` argument to the default parser, see :class:`eoe.datasets.MSM` """
    def combined_parser_modify(parser):
        parser.add_argument(
            '--ms-mode', type=str, default=(),  nargs='+',
            choices=['+'.join((i, j)) for i, j in product(TRANSFORMS.keys(), DS_PARTS.keys())],
            help="Defines a list of MSMs (multi-scale modes) to be used for training and testing. "
                 "An MSM consists of a transformation and a `type` that defines to which part of the data the "
                 "transformation will be applied. An MSM is defined by TRANSFORM+DSPART. "
                 "For instance, the MSM defined by `lpf+train_nominal` will apply a low pass filter "
                 "to normal training samples only. For more details have a look at :class:`eoe.datasets.MSM`. "
        )
        parser.add_argument(
            '--magnitude', type=int, default=14,
            help="Defines a magnitude for all MSMs (see `--ms-mode`). The meaning of magnitude may vary depending on the "
                 "transformation. For a description of the magnitude have a look at Appendix C in our paper or "
                 "`eoe.utils.transformations`. In general, the larger the magnitude the more severe the transformation. "
                 "Note that the multi-scale experiment (see :method:`multiscale_experiment`) ignores this argument since "
                 "it iterates over a range of magnitudes. "
        )
        if modify_parser is not None:
            modify_parser(parser)

    def combined_args_modify(args):
        args.ms_mode = [MSM(msm.split('+')[0], msm.split('+')[1], args.magnitude) for msm in args.ms_mode]
        if modify_args is not None:
            modify_args(args)

    return default_argsparse(modify_descr, combined_parser_modify, combined_args_modify)


def evolve_argsparse(modify_descr: Callable[[str, ], str], modify_parser: Callable[[ArgumentParser], None] = None,
                     modify_args: Callable[[Namespace], None] = None):
    """ adds arguments for finding optimal (best or worst) OE samples to the default parser """

    def evolve_descr_modify(s: str):
        s += "Repeats this complete procedure with different fixed Outlier Exposure (OE) subsets of the OE dataset. " \
             "The OE subsets are chosen according to an evolutionary algorithm that aims to optimize the average AUC " \
             "across the set of classes and random seeds. In the end, this algorithm will find the `best` OE samples. "
        if modify_descr is not None:
            s = modify_descr(s)
        return s

    def evolve_modify_parser(parser: ArgumentParser):
        parser.add_argument(
            '--ev-oesize', type=int, default=1,
            help="Determines the OE subset size."
        )
        parser.add_argument(
            '--ev-generation-pool', type=int, default=16,
            help="The pool size of the evolutionary algorithm. This is the size of different OE subsets per generation. "
                 "The algorithm selects, mutates, and mates these subsets to create a new generation. "
        )
        parser.add_argument(
            '--ev-mutation-pool', type=int, default=100,
            help="The pool size of the available mutations. If the algorithm decides to mutate an OE subset, it can "
                 "choose from this many random OE images to replace the ones in the subset. The replacement is the mutation. "
        )
        parser.add_argument(
            '--ev-mutation-indp', type=float, default=1.0,
            help="If the algorithm decides to mutate an OE subset, this determines the chance to mutate individual OE samples. "
                 "For instance, for 0.5, it would only replace/mutate roughly half of the individual OE samples of the "
                 "to-be-mutated OE subset. "
        )
        parser.add_argument(
            '--ev-mutation-oneofkbest', type=int, default=3,
            help="The final candidate pool size for mutation and mating. That is, from the initial mutation pool keep only "
                 "the k samples that have the least distance to the OE sample that is to be replaced. From this final pool size "
                 "randomly pick one for replacement. "
        )
        parser.add_argument(
            '--ev-mutation-chance', type=float, default=0.5,
            help="Determines the chance to mutate an OE subset for the next generation. For instance, for 0.5, this would "
                 "mutate approximately half of the OE subsets for the next generation. "
        )
        parser.add_argument(
            '--ev-mate-chance', type=float, default=0.2,
            help="Determines the chance to mate two OE subsets for the next generation. For instance, for 0.5, this would "
                 "mate approximately half of the OE subsets for the next generation. This action replaces only one of the two "
                 "parents. The other one can still be mutated or mated. "
        )
        parser.add_argument(
            '--ev-generations', type=int, default=30,
            help="The number of total generations. The algorithm terminates once the number is reached. "
        )
        parser.add_argument(
            '--ev-select-toursize', type=int, default=3,
            help="The tournament size. For each new generation, before mutating and mating, the algorithms selects new "
                 "oe subsets from the old generation according to a tournament selection rule. That is, it samples "
                 "tournament-size many samples randomly and picks the one with the best fitness (i.e., mean AUC). "
                 "The sampling works with replacement, so there can be duplicates in the new generation. "
        )
        parser.add_argument(
            '--ev-minimize-fitness', action='store_true',
            help="Activating this reverses the algorithm's optimization aim. That is, it seeks to minimize the "
                 "mean AUC of its OE subsets instead of maximizing it. "
        )
        parser.add_argument(
            '--ev-continue-run', type=str, default=None,
            help='Optional. If provided, needs to be a path to a logging directory of a previous evolve experiment. '
                 'Similar to `--load`, load the configuration. However, instead of loading model snapshots this will make '
                 'the script load the genealogical tree with all the fitness values (i.e., mean AUCs) and generations.'
                 'The script will continue where the previous experiment stopped. That is, it will take the latest completed '
                 'generation, create offsprings with them, and then continue with the evolutionary algorithm until '
                 'the total number of generations has been reached. Note that the `--load` parameter does not work '
                 'with the evolve experiments. It will simply be ignored. Also note that continuing an evolution experiment '
                 'like that requires some json files (e.g., containing the genealogical tree) that are logged '
                 'either after the training is completed or if an exception is caught. However, it is not logged if '
                 'the process is killed. In this case, continuing the experiment with `--ev-continue-run` will not work. '
        )
        if modify_parser is not None:
            modify_parser(parser)

    def evolve_modify_args(args):
        if args.load is not None:
            raise ValueError('For the evolutionary algorithm, `--load` has no impact. Use `--ev-continue-run` instead. ')
        if args.oe_size != np.infty:
            raise ValueError('For the evolutionary algorithm, `--oe-size` has no impact. Use `--ev-oesize` instead. ')
        if modify_args is not None:
            modify_args(args)

    return ms_argsparse(evolve_descr_modify, evolve_modify_parser, evolve_modify_args)


def create_trainer(trainer: str, comment: str, dataset: str, oe_dataset: str, epochs: int,
                   lr: float, wdk: float, milestones: List[int], batch_size: int, ad_mode: str,
                   gpus: List[int], model: torch.nn.Module, train_transform: Compose, val_transform: Compose,
                   oe_limit_samples: Union[int, List[int]] = np.infty, oe_limit_classes: int = np.infty,
                   msm: List[MSM] = (), logpath: str = None, **kwargs) -> ADTrainer:
    """
    This simply parses its parameters to create the correct trainer defined by the `trainer` str that defines the
    objective for the trainer. It also sets some additional parameters such as the datapath that defaults to
    `eoe/data` and creates a logger for the trainer. Returns the created trainer.
    For a description of the parameters have a look at :class:`eoe.training.ad_trainer.ADTrainer`.
    """
    datapath = pt.abspath(pt.join(__file__, '..', '..', '..', '..', 'data'))
    kwargs = dict(kwargs)
    superdir = kwargs.pop('superdir', '.')
    continue_run = kwargs.pop('continue_run', None)

    if continue_run is None:
        logger = Logger(pt.join(datapath, 'results', superdir) if logpath is None else logpath, comment)
    else:
        logger = Logger(continue_run + '---CNTD', noname=True)

    trainer = TRAINER[trainer](
        model, train_transform, val_transform, dataset, oe_dataset, pt.join(datapath, 'datasets'), logger,
        epochs, lr, wdk, milestones, batch_size, ad_mode, torch.device(gpus[0]),
        oe_limit_samples, oe_limit_classes, msm, **kwargs
    )
    return trainer


def evolve_trainer(trainer: str, comment: str, dataset: str, oe_dataset: str, epochs: int,
                   lr: float, wdk: float, milestones: List[int], batch_size: int, ad_mode: str,
                   gpus: List[int], model: torch.nn.Module, train_transform: Compose, val_transform: Compose,
                   oe_limit_samples: Union[int, List[int]] = np.infty, oe_limit_classes: int = np.infty,
                   msms: List[MSM] = (), evolve_tag: str = '', logpath: str = None,
                   classes: List[int] = None, **kwargs) -> Tuple[ADTrainer, VisionDataset, Logger]:
    """
    Similar to `create_trainer` but prepares a trainer suitable for the evolution experiment.
    To find optimal OE samples, the evolutionary algorithm repeatedly calls the trainer's `run` method
    that trains and evaluates a model for each class-seed combination.
    Though, finding best OE samples for multiple classes at once is not supported for now.
    The most important difference to the usual trainer is that the evolve trainer prepares the dataset only once to save time.
    This is also why multiple classes are not supported.
    Another difference is that it only logs the first training because otherwise it would create a spam of log directories.
    """
    evolve_dir = pt.join(kwargs.pop('superdir', '.'), f'log_{time_format(time.time())}_evolve_{evolve_tag}')
    continue_run = kwargs.pop('continue_run', None)
    if continue_run is not None:
        evolve_dir = continue_run + '---CNTD'
    trainer = create_trainer(
        trainer, comment, dataset, oe_dataset, epochs, lr, wdk, milestones, batch_size, ad_mode,
        gpus, model, train_transform, val_transform, oe_limit_samples, oe_limit_classes, msms, logpath,
        superdir=evolve_dir, continue_run=None, **kwargs
    )
    dummy = classes[0] if classes is not None and len(classes) > 0 else 0
    ds = load_dataset(
        trainer.dsstr, trainer.datapath, trainer.get_nominal_classes(dummy), 0,
        trainer.train_transform, trainer.test_transform, trainer.logger, trainer.oe_dsstr,
        trainer.oe_limit_samples, trainer.oe_limit_classes, trainer.msms
    )
    if classes is not None and len(classes) == 1:
        trainer.ds = ds
    else:
        raise NotImplementedError('Atm, evolve for multiple classes at once does not work.')
        # because trainer.run() in evolve.__init__.evaluate_individual() receives the classes and would recreate a dataset for
        # each class/seed combination since trainer.ds is None; but the individuals are set via
        # trainer.ds.oe.train_set.indices = ... and are thus removed; this would result
        # in complete nonsense results for the evolve experiment since for each individual (OE subset)
        # infinite OE samples are actually used for training!
        # Note that, in the paper in Table 3, we report the mean AUC of the best OE samples found for each class
        # and not the AUC of the best OE sample for all classes.
    ds = ds.oe.train_set
    logger = Logger(pt.join(trainer.logger.dir, '..'), noname=True)
    return trainer, ds, logger


def evolve_setup(oesize: int, generation_pool: int, mutation_pool: int, mutation_indp: float, mutation_oneofkbest: int,
                 mutation_chance: float, mate_chance: float, generations: int, select_toursize: int,
                 trainer: ADTrainer, oeds: VisionDataset,
                 args: Namespace, maxfit=True) -> Tuple[List[object], int, deap.base.Toolbox, dict, Tree]:
    """
    Prepares the evolutionary algorithm. The implementation is based on DEAP (https://github.com/deap/deap).
    @param oesize: See :method:`evolve_argsparse`.
    @param generation_pool: See :method:`evolve_argsparse`.
    @param mutation_pool: See :method:`evolve_argsparse`.
    @param mutation_indp: See :method:`evolve_argsparse`.
    @param mutation_oneofkbest: See :method:`evolve_argsparse`.
    @param mutation_chance: See :method:`evolve_argsparse`.
    @param mate_chance: See :method:`evolve_argsparse`.
    @param generations: See :method:`evolve_argsparse`.
    @param select_toursize: See :method:`evolve_argsparse`.
    @param trainer: The prepared ADTrainer instance (see :method:`evolve_trainer`).
    @param oeds: The prepared OE dataset.
    @param args: The trainer's configuration (the result of the argument parser).
    @param maxfit: See :method:`evolve_argsparse`. Corresponds to --ev-minimize-fitness.
    @return: a tuple of
        - the list of initial OE subsets (i.e., the first generation)
        - the generation id (i.e., 0 since it is the first)
        - the prepared DEAP toolbox that provides methods for mating, etc.
        - a dictionary containing the configuration of the evolutionary algorithm (including all the parameters of this method);
          also used to record the future fitness (i.e., mean AUC) of all the OE subsets of all generations and some statistics.
        - the genealogical tree that is used to record the relation of all OE subsets; that is, who are the parents
          and children of a sample. See :class:`eoe.evolve.tree.Tree`.
    """
    gen = 0
    history = {
        'pop': [], 'fit': [], 'mean_fit': [], 'std_fit': [], 'max_fit': [], 'min_fit': [],
        'setup': {
            'oesize': oesize, 'geneation_pool': generation_pool, 'mutation_pool': mutation_pool,
            'mutation_indp': mutation_indp, 'mutation_oneofkbest': mutation_oneofkbest,
            'mutation_chance': mutation_chance, 'mate_chance': mate_chance, 'generations': generations,
            'oeds': trainer.oe_dsstr, 'select_toursize': select_toursize
        }
    }

    if maxfit:
        creator.create('FitnessMax', base.Fitness, weights=(1.0, ))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    else:
        creator.create('FitnessMin', base.Fitness, weights=(-1.0, ))
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("init_individual", init_individual, oeds=oeds)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.init_individual, oesize)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', evaluate_individual, trainer=trainer, args=args, toolbox=toolbox)
    toolbox.register(
        "mate", mate_individuals, oeds=oeds, poolsize=mutation_pool, indp=mutation_indp, oneofkbest=mutation_oneofkbest,
    )
    toolbox.register(
        "mutate", mutate_individual, oeds=oeds, poolsize=mutation_pool, indp=mutation_indp, oneofkbest=mutation_oneofkbest,
    )
    toolbox.register("select", select_individual, tournsize=select_toursize)
    pop = toolbox.population(n=generation_pool)
    nodes = []
    for ind in pop:
        nodes.append(EvolNode(Individual(list(ind))))
        ind.file = None
    tree = Tree(*[node for node in nodes])
    return pop, gen, toolbox, history, tree


def rand_pick_setup(oesize: int, generation_pool: int,
                    trainer: ADTrainer, oeds: VisionDataset,
                    args: Namespace, maxfit=True) -> Tuple[List[object], int, deap.base.Toolbox, dict, Tree]:
    """
    Similar to `evolve_setup` but instead of an evolutionary algorithm prepares the DEAP toolbox to
    sample OE subsets completely randomly.
    @param oesize: The size of each OE subset.
    @param generation_pool: The number of randomly sampled OE subsets.
    @param trainer: The prepared ADTrainer instance.
    @param oeds: The prepared OE dataset.
    @param args: The trainer's configuration (the result of the argument parser).
    @param maxfit: See :method:`evolve_argsparse`. Corresponds to --ev-minimize-fitness.
    @return: a tuple of
        - the list of initial OE subsets (i.e., all the randomly sampled OE subsets)
        - the generation id (i.e., 0 since it is the first and only one in this case)
        - the prepared DEAP toolbox that provides methods for the ranomdly sampling
        - a dictionary containing the configuration of the random sampling (including all the parameters of this method);
          also used to record the future fitness (i.e., mean AUC) of all the OE subsets and some statistics.
        - the genealogical tree that is used to record the relation of all OE subsets;
          in this case this tree will have depth 1 and won't be of much use.
    """
    history = {
        'pop': [], 'fit': [], 'mean_fit': [], 'std_fit': [], 'max_fit': [], 'min_fit': [],
        'setup': {
            'oesize': oesize,
        }
    }

    if maxfit:
        creator.create('FitnessMax', base.Fitness, weights=(1.0, ))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    else:
        creator.create('FitnessMin', base.Fitness, weights=(-1.0, ))
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("init_individual", init_individual, oeds=oeds)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.init_individual, oesize)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', evaluate_individual, trainer=trainer, args=args, toolbox=toolbox)
    toolbox.register("mate", mate_individuals, oeds=oeds, poolsize=0, indp=0.0, oneofkbest=0,)
    toolbox.register("mutate", mutate_individual, oeds=oeds, poolsize=0, indp=0.0, oneofkbest=0,)
    toolbox.register("select", replace_individuals_randomly, oeds=oeds)
    pop = toolbox.population(n=generation_pool)
    nodes = []
    for ind in pop:
        nodes.append(EvolNode(Individual(list(ind))))
        ind.file = None
    tree = Tree(*[node for node in nodes])
    return pop, 0, toolbox, history, tree


def multiscale_experiment(args: Namespace, model: torch.nn.Module, train_transform: Compose, val_transform: Compose,
                          magnitudes: List[int] = (0, 1, 2, 4, 8, 16, 32), **kwargs):
    """
    Given: a list of multi-scale modes (MSMs) with different magnitudes.
    One MSM applies a data transformation on a certain data `type` only; `type` in this context means a certain part of
    the dataset such as `normal training samples` or `anomalous test samples`.
    For more details see :class:`eoe.datasets.MSM`.

    Create multiple trainers with each having a different magnitude for the list of MSMs.
    Apart from the MSM magnitudes, each trainer uses the same configuration.
    Then run all these trainers sequentially.
    In other words, this repeats the usual training for different filter magnitudes of the MSMs to
    investigate the effect of the magnitude. Results for such an experiment can be found in our paper in Appendix C.

    @param args: The trainer's configuration (the result of the argument parser). Also contains the list of MSMs.
    @param model: The prepared AD model.
    @param train_transform: The training data transformation pipeline.
    @param val_transform: The test/validation data transformation pipeline.
    @param magnitudes: A list of the different magnitudes. The length of this list defines the number of full training procedures.
    @param kwargs: other optional parameters such as one for continuing a multiscale experiment.
    """
    aucs = []
    superdir = kwargs.pop('superdir', '.')
    continue_run = kwargs.pop('continue_run', [])
    continue_last_magnitude = kwargs.pop('continue_last_magnitude', (None, None))
    plot_elsewhere = kwargs.pop('plot_elsewhere', None)
    datapath = pt.abspath(pt.join(__file__, '..', '..', '..', '..', 'data'))

    if len(continue_run) == 0:
        logger = Logger(pt.join(datapath, 'results', superdir), args.comment)
    else:
        logger = Logger(args.continue_run if plot_elsewhere is None else plot_elsewhere, noname=True)
    magn0_models = None
    for i, magnitude in enumerate(magnitudes):
        if len(continue_run) > i:
            aucs.append(continue_run[i])
            if all([msm.ds_part not in (TRAIN_NOMINAL_ID, TRAIN_OE_ID) for msm in args.ms_mode]):
                raise NotImplementedError('For blur test only, the magn0 models have to be loaded from the snapshots!')
            continue
        trainer = create_trainer(
            args.objective, f'magnitude_{magnitude}', args.dataset, args.oe_dataset,
            args.epochs, args.learning_rate, args.weight_decay,
            args.milestones, args.batch_size, args.ad_mode, args.devices, model, train_transform, val_transform,
            msm=[msm.set_magnitude(magnitude) for msm in args.ms_mode], logpath=logger.dir,
            oe_limit_samples=args.oe_size, continue_run=continue_last_magnitude[1], **kwargs
        )
        if magnitude != 0 and all([msm.ds_part not in (TRAIN_NOMINAL_ID, TRAIN_OE_ID) for msm in args.ms_mode]):
            trainer.epochs = 0
            _, results = trainer.run(args.classes, args.iterations, magn0_models)
        else:
            models, results = trainer.run(args.classes, args.iterations, continue_last_magnitude[0])
        continue_last_magnitude = (None, None)
        magn0_models = models if magnitude == 0 else magn0_models
        aucs.append((results['mean_auc'], results['std_auc']))

    print(f'----------------- {args.ms_mode} OVERVIEW -----------------')
    for s, (a, std) in zip(magnitudes, aucs):
        print(f'{args.ms_mode} with magnitude={s:02d} yielded {a*100:04.2f} +- {std*100:04.2f}.')
    results = {
        'magnitudes': magnitudes, 'aucs': list(zip(*aucs))[0], 'stds': list(zip(*aucs))[1],
        'classes': args.classes, 'comment': args.comment, 'ms_mode': [repr(msm) for msm in args.ms_mode],
        'dataset': args.dataset
    }
    logger.logjson('results', results)


def load_setup(path: str, args: Namespace, check_train_transform: Compose,
               check_val_transform: Compose) -> Tuple[List[List[str]], str]:
    """
    Loads the setup/configuration from given path, including all model snapshots.
    Can be used to repeat or continue a previous experiment.
    @param path: the path to the logging directory of the experiment from which the configuration is to be loaded.
    @param args: the args namespace where the setup is to be loaded to.
    @param check_train_transform: since the transforms cannot be automatically loaded,
        check if their logged string representation matches this parameter's string representation.
    @param check_val_transform: since the transforms cannot be automatically loaded,
        check if their logged string representation matches this parameter's string representation.
    @return: a tuple of
        - a list (len = #classes) with each element again being a list (len = #seeds) of filepaths to model snapshots.
          Some can be None. The snapshots may also contain the training state such as the last epoch trained.
        - the path from which the configuration was loaded.
    """
    if path is None:
        return None, None
    elif path.startswith('sftp://'):  # 7 chars
        path = path[7:][path[7:].index('/'):]  # sft://foo@bar.far.com/PATH -> /PATH
    print(f'Load setup from {path}')
    with open(pt.join(path, 'setup.json'), 'r') as reader:
        setup = json.load(reader)
    with open(pt.join(path, 'setup_v1.json'), 'r') as reader:
        setup.update(json.load(reader))
    assert [x.replace("'normalize'", 'normalize') for x in json.loads(json.dumps(check_train_transform, cls=SetupEncoder))] == \
           setup.pop('train_transform'), \
           f'The loaded train transformation string representation does not match the set one. Please match manually. '
    assert [x.replace("'normalize'", 'normalize') for x in json.loads(json.dumps(check_val_transform, cls=SetupEncoder))] == \
           setup.pop('test_transform'), \
           f'The loaded test transformation string representation does not match the set one. Please match manually. '
    assert setup.pop('oe_limit_classes') == np.infty
    setup_load = setup.pop('load')
    assert setup_load is None or all([isinstance(seed, str) or seed is None for cls in setup_load for seed in cls])
    assert setup.pop('dataset') == args.dataset, \
        f'It seems like the set dataset ({args.dataset}) is not the one found in the loaded experiment. Please match manually. '
    assert f'_{args.objective}_' in path, \
        f'It seems like the set objective ({args.objective}) is not the one found in the loaded experiment. ' \
        f'Please match manually. '
    args.oe_dataset = setup.pop('oe_dataset')
    args.epochs = setup.pop('epochs')
    args.learning_rate = setup.pop('lr')
    args.weight_decay = setup.pop('wdk')
    args.milestones = setup.pop('milestones')
    args.batch_size = setup.pop('batch_size')
    args.ad_mode = setup.pop('ad_mode')
    args.oe_size = setup.pop('oe_limit_samples', np.infty)
    args.ms_mode = setup.pop('msms', None)
    if args.ms_mode is not None:
        args.ms_mode = [MSM.load(msm) for msm in args.ms_mode]
    args.model = setup.pop('model', None)
    args.classes = setup.pop('run_classes')
    args.iterations = setup.pop('run_seeds')
    setup.pop('workers')
    setup.pop('device')
    setup.pop('datapath')
    setup.pop('logger')
    assert len(setup) == 0, f'There are unexpected arguments in the loaded setup: {setup.keys()}.'
    classes = args.classes if args.classes is not None else range(no_classes(args.dataset))
    snapshots = []
    for c in range(no_classes(args.dataset)):
        snapshots.append([])
        for i in range(args.iterations):
            if c in classes:
                snapshot = pt.join(path, 'snapshots', f'snapshot_cls{c}_it{i}.pt')
                if not pt.exists(snapshot):
                    snapshot = None
                snapshots[-1].append(snapshot)
                if setup_load is not None:
                    if setup_load[c][i] is not None:
                        # print('Overwriting snapshot loaded in previous training with the snapshot of previous training.')
                        pass
            else:
                snapshots[-1].append(None)
    return snapshots, path


def load_evolve(path: str, tree: Tree, history: dict, pop: List[object], gen: int, logger: Logger, trainer: ADTrainer,
                dataset: VisionDataset, args: Namespace, check_train_transform: Compose,
                check_val_transform: Compose) -> Tuple[dict, List[object], int, Logger, deap.base.Toolbox]:
    """
    Similar to :method:`load_setup` but loads the configuration of a previous evolution experiment.
    Note that `load_setup` should be called first to retrieve a trainer and dataset with the correct configuration.

    @param path: the path to the logging directory of the evolve experiment from which the configuration is to be loaded.
    @param tree: the prepared (empty) tree instance into which the logged tree from the previous
        evolution experiment is loaded. See :class:`eoe.evolve.tree.Tree` and :method:`evolve_setup`.
        If the tree is not empty it will be overwritten.
    @param history: the dictionary into which the history dictionary of the previous evolution experiment is loaded;
        the history records the fitness (i.e., mean AUC) and configuration of all OE subsets.
        If the given history is not empty it will be overwritten by the history of the to-be-loaded experiment.
    @param pop: the current population, will be overwritten with the latest generation of the to-be-loaded experiment.
    @param gen: the current generation number, will be overwritten with the
        latest generation number of the to-be-loaded experiment.
    @param logger: the prepared logger. Its logging directory should have already been set to the correct directory that
        contains the `---CNTD` suffix via :method:`evolve_trainer` by passing the `continue_run` parameter in **kwargs.
    @param trainer: the prepared trainer. Should have already been updated with the loaded training configuration
        via, e.g., :method:`load_setup`. It will be used to update the DEAP toolbox with the loaded configuration.
    @param dataset: the prepared dataset. Should have already been updated with the loaded training configuration
        via, e.g., :method:`load_setup`. It will be used to update the DEAP toolbox with the loaded configuration.
    @param args: the args namespace where the setup is to be loaded to.
    @param check_train_transform: since the transforms cannot be automatically loaded,
        check if their logged string representation matches this parameter's string representation.
    @param check_val_transform: since the transforms cannot be automatically loaded,
        check if their logged string representation matches this parameter's string representation.
    @return: a tuple of
        - a history dictionary containing the loaded configuration of the evolutionary algorithm;
          also contains the recorded fitness of all OE subsets; will be used to continue recording the fitness of further subsets.
        - the list of the latest generation; i..e, OE subsets
        - the generation number of the loaded latest generation
        - the logger
        - the updated DEAP toolbox that provides methods for mating, etc.
    """
    if path is None:
        return history, pop, gen, logger, None

    print(f'Load evolve setup from {path}')
    with open(pt.join(path, 'results.json'), 'r') as reader:
        loaded_history = json.load(reader)
    setup = deepcopy(loaded_history['setup'])
    for k, v in setup.items():
        assert k in history['setup'], f"{k} is in loaded setup but not in setup!"
    for k, v in history['setup'].items():
        assert k in setup, f"{k} is in setup but not in loaded setup!"
    args.ev_oesize = setup.pop('oesize')
    args.ev_generation_pool = setup.pop('geneation_pool')
    args.ev_mutation_pool = setup.pop('mutation_pool')
    args.ev_mutation_indp = setup.pop('mutation_indp')
    args.ev_mutation_oneofkbest = setup.pop('mutation_oneofkbest')
    args.ev_mutation_chance = setup.pop('mutation_chance')
    args.ev_mate_chance = setup.pop('mate_chance')
    args.ev_generations = setup.pop('generations')
    args.ev_oeds = setup.pop('oeds')
    args.ev_select_toursize = setup.pop('select_toursize')
    assert len(setup) == 0, f"setup contains unexpected keys ({setup.keys()})"
    history = loaded_history

    load_setup(
        pt.join(path, [d for d in os.listdir(path) if pt.isdir(pt.join(path, d)) and d.startswith("log_")][0]),
        args, check_train_transform, check_val_transform
    )

    tree.load(pt.join(path, 'evolution.json'))
    nodes = tree.bfs()[1:]
    if len(history['pop']) > 0:
        nodes = [[n for n in nodes if n.content.values == val][0].content for val in history['pop'][-1]]
    else:
        raise ValueError(f'The loaded experiment at {path} has no finished generation that can be used to continue it.')
    pop = []
    for n in nodes:
        pop.append(creator.Individual(n.values))
        pop[-1].fitness.values = (n.fitness, )
        pop[-1].file = n.file
    gen = len(history['pop'])

    _, _, toolbox, _, _ = evolve_setup(
        args.ev_oesize, args.ev_generation_pool, args.ev_mutation_pool, args.ev_mutation_indp, args.ev_mutation_oneofkbest,
        args.ev_mutation_chance, args.ev_mate_chance, args.ev_generations, args.ev_select_toursize, trainer, dataset, args,
        not args.ev_minimize_fitness
    )

    return history, pop, gen, logger, toolbox

