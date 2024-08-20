import sys
import inspect

import torchvision.transforms as transforms
from torchvision.transforms import Compose

import eoe.models.custom as custom_models_pck
from eoe.main import default_argsparse, create_trainer, load_setup
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

if __name__ == '__main__':
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
            help="A path to the custom dataset's test data directory. "
                 "The directory has to contain at least one of the following folders: "
                 "(1) 'normal' for normal test samples, "
                 "(2) 'anomalous' for anomalous test samples, and "
                 "(3) 'unlabeled' for unlabeled test samples. "
                 "One of these folders needs to be non-empty. "
                 "If both (1) and (2) contain each at least one image, an AuROC will be computed for (1) vs (2)."
        )
        group.add_argument(
            '--log-path', type=str, required=True,
            help="A path to a directory where results are to be logged (including snapshots, etc.)."
        )
        group.add_argument(
            '--custom-model-snapshot', type=str, metavar='FILE-PATH', required=True,
            help="Path to a previously EOE-trained model snapshot. "
                 "Such a snapshot is automatically created during training with EOE and is located at the log directory in the "
                 "subfolder 'snapshots' for each iteration. The snapshot's model weights need to match the chosen "
                 "--custom-model-name. "
        )
        group.add_argument(
            '--custom-model-name', type=str, choices=list(custom_models.keys()), default="WideResNetCustom",
            help="The class name of any model implemented in :file:`xad.models.custom`."
        )
        group.add_argument(
            '--custom-model-add-prediction-head', action='store_true',
            help="Adds a randomly-initialized prediction head with either "
                 "256 output neurons (HSC, ...) or 1 neuron (BCE, focal, ...) to the custom model."
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
                 "this sets the target height and width of the center crop transformation. "
                 "If either height or width are less or equal to 0, no center crop will be used. "
        )
        parser.set_defaults(
            comment='{obj}_custom_inference',
            objective='hsc',
            dataset='custom',
            oe_dataset=None,
            epochs=150,
            learning_rate=1e-3,
            weight_decay=0,
            milestones=[100, 125],
            batch_size=128,
            devices=[0],
            classes=None,
            iterations=1
        )

    def modify_args(args):
        if args.dataset is not None and args.dataset != "custom":
            raise ValueError(f"The argument dataset needs to be 'custom' for custom training.")
        if args.load is not None:
            raise NotImplementedError(f"Continuing an experiment for custom training is not supported at the moment.")
        if args.classes is not None:
            raise ValueError(f"The argument classes is not supported for custom training.")
        args.dataset = 'custom'
        args.iterations = 1  # there's no point in evaluating multiple times
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
    ADCustomDS.eval_only = True
    ADCustomDS.base_folder = "."

    args.comment = args.comment.format(
        obj=args.objective, admode='',
    )
    if all([c > 0 for c in args.image_crop]):
        # change this to use different data transforms for inference when args.image_crop contains positive numbers only
        train_transform = val_transform = Compose([
            transforms.Resize(args.image_resize),
            transforms.CenterCrop(args.image_crop),
            transforms.ToTensor(),
            'normalize'
        ])
    else:
        # change this to use different data transforms for inference when args.image_crop contains any non-positive number
        train_transform = val_transform = Compose([
            transforms.Resize(args.image_resize),
            transforms.ToTensor(),
            'normalize'
        ])
    model = custom_models[args.custom_model_name](
        prediction_head=args.custom_model_add_prediction_head,
        clf=args.objective in ('bce', 'focal'),
    )

    print('Program started with:\n', vars(args))
    trainer = create_trainer(
        args.objective, args.comment, args.dataset, args.oe_dataset, args.epochs, args.learning_rate, args.weight_decay,
        args.milestones, args.batch_size, args.ad_mode, args.devices, model, train_transform, val_transform,
        oe_limit_samples=args.oe_size, dataset_path=args.custom_dataset_path, logpath=args.log_path
    )

    results = trainer.run([0], args.iterations, [[args.custom_model_snapshot] * args.iterations], train=False)

