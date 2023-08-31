import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import Compose

from eoe.main import default_argsparse, create_trainer, load_setup
from eoe.models.resnet import WideResNet
from eoe.datasets import DS_CHOICES
from eoe.datasets.custom import ADCustomDS

DS_CHOICES['custom'] = {
    'class': ADCustomDS,  # static, don't change this
    'default_size': 256,  # can be set via arguments
    'no_classes': -1,  # is automatically extracted for custom datasets, thus can be ignored
    'oe_only': False,  # static, don't change this
    'str_labels': []  # is automatically extracted for custom datasets, thus can be ignored
}

if __name__ == '__main__':
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
        
                The same holds for the test set, where "/train/" has to be replaced by "/test/", and in (2) the anomalies are not 
                used as OE but as ground-truth anomalies for testing.
            """
        )
        parser.set_defaults(
            comment='{obj}_custom{admode}_{oelimit}',
            objective='hsc',
            dataset='custom',
            oe_dataset='imagenet21k',
            epochs=150,
            learning_rate=1e-3,
            weight_decay=0,
            milestones=[100, 125],
            batch_size=128,
            devices=[0],
            classes=None,
            iterations=10
        )
    args = default_argsparse(
        lambda s: f"{s} This specific script comes with a default configuration for custom datasets.",
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
        oelimit=f'_OE{args.oe_size}' if args.oe_size < np.infty else ''
    )
    train_transform = transforms.Compose([  # change this to use different data transforms for training
        transforms.Resize(256),
        transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
        transforms.RandomHorizontalFlip(p=0.5),
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
    snapshots, continue_run = load_setup(args.load, args, train_transform, val_transform)
    model = WideResNet(clf=args.objective in ('bce', 'focal'))  # change this line for a different model

    print('Program started with:\n', vars(args))
    trainer = create_trainer(
        args.objective, args.comment, args.dataset, args.oe_dataset, args.epochs, args.learning_rate, args.weight_decay,
        args.milestones, args.batch_size, args.ad_mode, args.devices, model, train_transform, val_transform,
        oe_limit_samples=args.oe_size, continue_run=continue_run, superdir=args.superdir
    )

    trainer.run(args.classes, args.iterations, snapshots)
