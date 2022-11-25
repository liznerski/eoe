import os.path as pt
from argparse import Namespace
from typing import Tuple

import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose

from eoe.evolve import evaluate
from eoe.main import ms_argsparse, rand_pick_setup, evolve_trainer
from eoe.models.cnn import CNN32
from eoe.training.ad_trainer import ADTrainer
from eoe.utils.logger import Logger


def init() -> Tuple[ADTrainer, VisionDataset, Namespace, Logger]:
    def modify_parser(parser):
        parser.add_argument(
            '--ev-oesize', type=int, default=1, help='The size of the OE subsets. '
        )
        parser.add_argument(
            '--ev-samples', type=int, default=2000, help='The number of OE subsets that are randomly chosen.'
        )
        parser.set_defaults(
            comment='RANDPICK_{obj}_cifar10_cl{classes}_its{its}',
            objective='hsc',
            dataset='cifar10',
            oe_dataset='tinyimages',
            epochs=30,
            learning_rate=1e-3,
            weight_decay=0,
            milestones=[25],
            batch_size=128,
            devices=[0],
            classes=[0],
            iterations=2,
        )
    args = ms_argsparse(
        lambda s: f"{s} Repeats this complete procedure with different fixed Outlier Exposure (OE) subsets of the OE dataset. "
                  f"The OE subsets are chosen randomly. "
                  f"This specific script comes with a default configuration "
                  f"for finding OE subsets for CIFAR-10 for class 0 being normal.",
        modify_parser,
    )
    if args.oe_size != np.infty:
        raise ValueError('For finding random OE subsets, `--oe-size` has no impact. Use `--ev-oesize` instead. ')
    args.comment = args.comment.format(
        obj=args.objective, admode=args.ad_mode, classes=args.classes, its=args.iterations,
    )
    train_transform = transforms.Compose([
        # transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x + 0.001 * torch.randn_like(x)),
        'normalize'
    ])
    val_transform = Compose([
        transforms.ToTensor(),
        'normalize'
    ])
    model = CNN32(bias=True, clf=args.objective in ('bce', 'focal'))

    print('Program started with:\n', vars(args))
    trainer, oeds, logger = evolve_trainer(
        args.objective, '', args.dataset, args.oe_dataset, args.epochs, args.learning_rate, args.weight_decay,
        args.milestones, args.batch_size, args.ad_mode, args.devices, model, train_transform, val_transform,
        evolve_tag=args.comment, classes=args.classes, msms=args.ms_mode, superdir=args.superdir,
        # continue_run=args.ev_continue_run
    )
    return trainer, oeds, args, logger


if __name__ == '__main__':
    trainer, ds, args, logger = init()
    pop, start_gen, toolbox, history, tree = rand_pick_setup(args.ev_oesize, args.ev_samples, trainer, ds, args)

    try:
        evaluate(pop, pop, start_gen, toolbox, history, tree, ds, logger)
    finally:
        logger.logjson('results', history)
        tree.save(pt.join(logger.dir, 'evolution'))
        tree.imsave_collection_best(logger, args.ms_mode)

    print()
