import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import Compose

from eoe.main import default_argsparse, create_trainer, load_setup
from eoe.models.resnet import WideResNet

if __name__ == '__main__':
    def modify_parser(parser):
        parser.set_defaults(
            comment='{obj}_imagenet_{admode}{oelimit}',
            objective='hsc',
            dataset='imagenet',
            oe_dataset='imagenet21k',
            epochs=30,
            learning_rate=1e-3,
            weight_decay=0,
            milestones=[25],
            batch_size=128,
            devices=[0],
            classes=None,
            iterations=2
        )
    args = default_argsparse(
        lambda s: f"{s} This specific script comes with a default configuration for training ImageNet-30 "
                  f"fast (i.e., less epochs and augmentations).", modify_parser
    )
    args.comment = args.comment.format(
        obj=args.objective, admode=args.ad_mode, oelimit=f'_OE{args.oe_size}' if args.oe_size < np.infty else ''
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
    snapshots, continue_run = load_setup(args.load, args, train_transform, val_transform)
    model = WideResNet(clf=args.objective in ('bce', 'focal'))

    print('Program started with:\n', vars(args))
    trainer = create_trainer(
        args.objective, args.comment, args.dataset, args.oe_dataset, args.epochs, args.learning_rate, args.weight_decay,
        args.milestones, args.batch_size, args.ad_mode, args.devices, model, train_transform, val_transform,
        oe_limit_samples=args.oe_size, continue_run=continue_run, superdir=args.superdir
    )

    trainer.run(args.classes, args.iterations, snapshots)
