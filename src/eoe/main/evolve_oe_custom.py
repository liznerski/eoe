import os
import os.path as pt
from argparse import Namespace
from typing import Tuple

import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose

from eoe.datasets import DS_CHOICES
from eoe.datasets.custom import ADCustomDS
from eoe.evolve import evolve, evaluate
from eoe.main import evolve_argsparse, evolve_setup, load_evolve, load_setup, evolve_trainer
from eoe.models.resnet import WideResNet
from eoe.training.ad_trainer import ADTrainer
from eoe.utils.logger import Logger

DS_CHOICES['custom'] = {
    'class': ADCustomDS,  # static, don't change this
    'default_size': 256,  # can be set via arguments
    'no_classes': -1,  # is automatically extracted for custom datasets, thus can be ignored
    'oe_only': False,  # static, don't change this
    'str_labels': []  # is automatically extracted for custom datasets, thus can be ignored
}


def init() -> Tuple[ADTrainer, VisionDataset, Namespace, Logger]:
    def modify_parser(parser):
        group = parser.add_argument_group('custom-dataset')
        group.add_argument(
            '--custom-dataset-default-size', type=int, default=256,
            help="The custom dataset's default size, "
                 "which should equal the size of the first resize in the transforms pipeline."
        )
        group.add_argument(
            '--custom-dataset-ovr', action='store_true', default=False,
            help="""
                Determines whether this dataset expects the folder to be of the form specified in (1), which follows the
                one-vs-rest approach, or of the form specified in (2), which follows the general AD approach.
                
                The data is expected to be contained in class folders. We distinguish between
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
                root/custom/train/hazelnut/anomalous/xxa.png    -- may be used during training as OE with --oe-dataset custom 
        
                root/custom/train/screw/normal/123.png
                root/custom/train/screw/normal/nsdf3.png
                root/custom/train/screw/anomalous/asd932_.png   -- may be used during training as OE with --oe-dataset custom 
        
                The same holds for the test set, where "train" has to be replaced by "test" and the anomalies are not 
                used as OE but as ground-truth anomalies for testing.
            """
        )
        parser.set_defaults(
            comment='{obj}_custom_cl{classes}',
            objective='hsc',
            dataset='custom',
            oe_dataset='imagenet21k',
            epochs=30,
            learning_rate=1e-3,
            weight_decay=0,
            milestones=[25],
            batch_size=128,
            devices=[0],
            classes=[0],
            iterations=2,
            ev_oesize=1,
            ev_generation_pool=64,
            ev_mutation_pool=10000,
            ev_mutation_indp=1,
            ev_mutation_oneofkbest=50,
            ev_mutation_chance=0.55,
            ev_mate_chance=0.05,
            ev_generations=50,
            ev_select_toursize=3,
        )
    args = evolve_argsparse(
        lambda s: f"{s} This specific script comes with a default configuration "
                  f"for finding optimal OE subsets for custom datasets for class 0 being normal.",
        modify_parser
    )
    if args.ad_mode != 'one_vs_rest':
        raise ValueError(
            f"The AD mode is changed to {args.ad_mode}. Note that custom datasets ignore the AD mode. "
            f"The mode is instead set via --custom-dataset-ovr."
        )
    DS_CHOICES['custom']['default_size'] = args.custom_dataset_default_size
    ADCustomDS.ovr = args.custom_dataset_ovr

    args.comment = args.comment.format(
        obj=args.objective, admode='_one_vs_rest' if args.custom_dataset_ovr else '',
        classes="+".join((str(c) for c in args.classes)), its=args.iterations,
    )
    train_transform = transforms.Compose([  # change this to use different data transforms for training
        transforms.Resize(256),
        # transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        'normalize'
    ])
    val_transform = Compose([  # change this to use different data transforms for testing
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        'normalize'
    ])
    if args.ev_continue_run is not None:
        training_log_dirs = [
            pt.join(args.ev_continue_run, f) for f in os.listdir(args.ev_continue_run)
            if pt.isdir(pt.join(args.ev_continue_run, f)) and f.startswith('log_')
        ]
        if len(training_log_dirs) == 0:
            raise ValueError(f"Could not find a training log directory in {args.ev_continue_run}.")
        _, continue_run = load_setup(training_log_dirs[0], args, train_transform, val_transform)
    model = WideResNet(clf=args.objective in ('bce', 'focal'))  # change this line for a different model

    print('Program started with:\n', vars(args))
    trainer, oeds, logger = evolve_trainer(
        args.objective, '', args.dataset, args.oe_dataset, args.epochs, args.learning_rate, args.weight_decay,
        args.milestones, args.batch_size, args.ad_mode, args.devices, model, train_transform, val_transform,
        evolve_tag=args.comment, classes=args.classes, msms=args.ms_mode, superdir=args.superdir,
        continue_run=args.ev_continue_run
    )
    return trainer, oeds, args, logger


if __name__ == '__main__':
    trainer, ds, args, logger = init()
    pop, start_gen, toolbox, history, tree = evolve_setup(
        args.ev_oesize, args.ev_generation_pool, args.ev_mutation_pool, args.ev_mutation_indp, args.ev_mutation_oneofkbest,
        args.ev_mutation_chance, args.ev_mate_chance, args.ev_generations, args.ev_select_toursize, trainer, ds, args,
        not args.ev_minimize_fitness
    )
    if args.ev_continue_run is not None:
        history, pop, start_gen, logger, toolbox = load_evolve(
            args.ev_continue_run, tree, history, pop, start_gen, logger, trainer, ds,
            args, trainer.train_transform, trainer.test_transform
        )

    try:
        if start_gen == 0:
            evaluate(pop, pop, start_gen, toolbox, history, tree, ds, logger)
            start_gen += 1
        for gen in range(start_gen, args.ev_generations):
            evolve(pop, gen, toolbox, args.ev_mate_chance, args.ev_mutation_chance, history, tree, ds, logger)
    finally:
        logger.logjson('results', history)
        tree.save(pt.join(logger.dir, 'evolution'))
        tree.imsave_collection_best(logger, args.ms_mode)

    print()
