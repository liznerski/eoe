import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import Compose

from eoe.main import default_argsparse, create_trainer, load_setup
from eoe.models.cnn import CNN28

if __name__ == '__main__':
    def modify_parser(parser):
        parser.set_defaults(
            comment='{obj}_fmnist_{admode}{oelimit}',
            objective='hsc',
            dataset='fmnist',
            oe_dataset='cifar100',
            epochs=200,
            learning_rate=1e-3,
            weight_decay=0,
            milestones=[100, 150],
            batch_size=128,
            devices=[0],
            classes=None,
            iterations=5,
        )
    args = default_argsparse(
        lambda s: f"{s} This specific script comes with a default configuration for Fashion-MNIST.", modify_parser
    )
    args.comment = args.comment.format(
        obj=args.objective, admode=args.ad_mode, oelimit=f'_OE{args.oe_size}' if args.oe_size < np.infty else ''
    )
    train_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(28, padding=3),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + 0.001 * torch.randn_like(x)),
        'normalize'
    ])
    val_transform = Compose([
        transforms.ToTensor(),
        'normalize'
    ])
    snapshots, continue_run = load_setup(args.load, args, train_transform, val_transform)
    model = CNN28(bias=True, clf=args.objective in ('bce', 'focal'))

    print('Program started with:\n', vars(args))
    trainer = create_trainer(
        args.objective, args.comment, args.dataset, args.oe_dataset, args.epochs, args.learning_rate, args.weight_decay,
        args.milestones, args.batch_size, args.ad_mode, args.devices, model, train_transform, val_transform,
        oe_limit_samples=args.oe_size, continue_run=continue_run, superdir=args.superdir
    )

    trainer.run(args.classes, args.iterations, snapshots)
