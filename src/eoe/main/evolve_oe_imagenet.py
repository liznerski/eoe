import os
import os.path as pt
from argparse import Namespace
from typing import Tuple

import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose

from eoe.evolve import evolve, evaluate
from eoe.main import evolve_argsparse, evolve_trainer, evolve_setup, load_evolve, load_setup
from eoe.models.resnet import WideResNet
from eoe.training.ad_trainer import ADTrainer
from eoe.utils.logger import Logger


def init() -> Tuple[ADTrainer, VisionDataset, Namespace, Logger]:
    def modify_parser(parser):
        parser.set_defaults(
            comment='{obj}_imagenet_cl{classes}',
            objective='hsc',
            dataset='imagenet',
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
                  f"for finding optimal OE subsets for ImageNet-30 for class 0 being normal.",
        modify_parser,
    )
    args.comment = args.comment.format(
        obj=args.objective, admode=args.ad_mode, classes="+".join((str(c) for c in args.classes)), its=args.iterations,
    )
    train_transform = transforms.Compose([
        transforms.Resize(256),
        # transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x + 0.001 * torch.randn_like(x)),
        'normalize'
    ])
    val_transform = Compose([
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
    model = WideResNet(clf=args.objective in ('bce', 'focal'))

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
