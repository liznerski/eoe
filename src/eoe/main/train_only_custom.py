import sys
import inspect

import numpy as np
import torchvision.transforms as transforms

import eoe.models.custom as custom_models_pck
from eoe.main import default_argsparse, create_trainer
from eoe.models.custom_base import CustomNet
from eoe.datasets import DS_CHOICES
from eoe.datasets.custom import ADCustomDS

DS_CHOICES['custom'] = {
    'class': ADCustomDS,  # static, don't change this
    'default_size': 256,  # can be set via arguments
    'no_classes': -1,  # is automatically extracted for custom datasets, thus can be ignored
    'oe_only': False,  # static, don't change this
    'str_labels': []  # is automatically extracted for custom datasets, thus can be ignored
}


def run():
    custom_models = {
        clz_name: clz for clz_name, clz in inspect.getmembers(custom_models_pck)
        if isinstance(clz, type) and issubclass(clz, CustomNet) and clz != CustomNet
    }

    def modify_parser(parser):
        group = parser.add_argument_group('custom-dataset')
        group.add_argument(
            '--custom-dataset-default-size', type=int, default=256,
            help="The custom dataset's default size. "
                 "The AD datasets use this to determine the dataset's statistics (mean, std) "
                 "if the data transformation pipelines contains no resize. "
        )
        group.add_argument(
            '--custom-dataset-path', type=str, metavar='DIRECTORY-PATH', required=True,
            help="A path to the custom dataset's training data. "
                 "The directory has to contain a folder named 'normal' for normal training samples."
                 "Additionally, it can contain a folder named 'anomalous' for anomalous training samples. "
                 "Both these folder have to contain images only."
        )
        group.add_argument(
            '--log-path', type=str, required=True,
            help="A path to a directory where results are to be logged (including snapshots, etc.)."
        )
        group.add_argument(
            '--custom-model-snapshot', type=str, metavar='FILE-PATH', default=None,
            help="A path to a snapshot. "
                 "The snapshot can either be:"
                 "(1) a state_dict of the feature model specified with --custom-model-name. In this case, "
                 "the feature model gets initialized with those weights. "
                 "(2) a snapshot that is automatically logged via previous EOE experiments. In this case, "
                 "the states of the model, optimizer, scheduler, and epoch are loaded. EOE continues training. "
        )
        group.add_argument(
            '--custom-model-name', type=str, choices=list(custom_models.keys()), default="WideResNetCustom",
            help="The class name of any model implemented in :file:`xad.models.custom`."
        )
        group.add_argument(
            '--custom-model-add-prediction-head', action='store_true',
            help="Adds a randomly-initialized prediction head with either "
                 "256 output neurons (HSC, ...) or 1 neuron (BCE, focal, ...) to the model."
        )
        group.add_argument(
            '--custom-model-freeze', action='store_true',
            help="Freezes gradients for a part of the model, depending on the implementation "
                 "of the model's self.freeze_parts() method. Per default, if argument is set, "
                 "freezes the entire feature extraction module."
        )
        group.add_argument(
            '--image-resize', type=int, default=(256, ), nargs='*',
            help="When using the default data transformation pipeline, "
                 "this sets the target size of the resize transformation. --image-resize accepts one or two integers. "
                 "If there are two, they define the target height and width. If there is one, the smaller edge "
                 "will be matched to this number. "
        )
        group.add_argument(
            '--image-crop', type=int, default=(224, 224), nargs=2,
            help="When using the default data transformation pipeline, "
                 "this sets the target height and width of the random crop transformation. "
                 "If either height or width are less or equal to 0, no random crop will be used. "
        )
        parser.set_defaults(
            comment='{obj}_custom_training',
            objective='hsc',
            dataset='custom',
            oe_dataset='custom',  # custom uses the folder "anomalous" of the dataset. Alternatively, use imagenet21k or others.
            epochs=150,
            learning_rate=1e-3,
            weight_decay=0,
            milestones=[100, 125],
            batch_size=128,
            devices=[0],
            classes=None,
            iterations=10
        )

    def modify_args(args):
        if args.dataset is not None and args.dataset != "custom":
            raise ValueError(f"The argument dataset needs to be 'custom' for custom training.")
        if args.load is not None:
            raise NotImplementedError(f"Continuing an experiment for custom training is not supported at the moment.")
        if args.classes is not None:
            raise ValueError(f"The argument classes is not supported for custom training.")
        args.dataset = 'custom'
        if args.ad_mode != 'one_vs_rest':
            print(
                f"The AD mode is changed to {args.ad_mode}. Custom datasets ignore the AD mode. ",
                file=sys.stderr
            )
        if len(args.image_resize) > 2:
            raise ValueError(
                f"--image-resize accepts one or two integers, but {len(args.image_resize)} ({args.image_resize}) are given. "
                f"If there are two integers, they define the target height and width. "
                f"If there is one, the smaller edge will be matched to this number."
            )

    args = default_argsparse(
        lambda s: f"{s} This specific script comes with a default configuration for custom datasets.",
        modify_parser, modify_args
    )
    DS_CHOICES['custom']['default_size'] = args.custom_dataset_default_size
    ADCustomDS.train_only = True
    ADCustomDS.base_folder = "."

    args.comment = args.comment.format(
        obj=args.objective, admode='',
        oelimit=f'_OE{args.oe_size}' if args.oe_size < np.infty else ''
    )
    if all([c > 0 for c in args.image_crop]):
        # change this to use different data transforms for training when args.image_crop contains positive numbers only
        train_transform = val_transform = transforms.Compose([
            transforms.Resize(args.image_resize),
            transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(args.image_crop),
            transforms.ToTensor(),
            'normalize'
        ])
    else:
        # change this to use different data transforms for training when args.image_crop contains any non-positive number
        train_transform = val_transform = transforms.Compose([
            transforms.Resize(args.image_resize),
            transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            'normalize'
        ])
    model = custom_models[args.custom_model_name](
        prediction_head=args.custom_model_add_prediction_head,
        clf=args.objective in ('bce', 'focal'),
        freeze=args.custom_model_freeze,
    )

    print('Program started with:\n', vars(args))
    trainer = create_trainer(
        args.objective, args.comment, args.dataset, args.oe_dataset, args.epochs, args.learning_rate, args.weight_decay,
        args.milestones, args.batch_size, args.ad_mode, args.devices, model, train_transform, val_transform,
        oe_limit_samples=args.oe_size, dataset_path=args.custom_dataset_path, logpath=args.log_path
    )
    trainer.run([0], args.iterations, [[args.custom_model_snapshot] * args.iterations], test=False)


if __name__ == '__main__':
    run()
