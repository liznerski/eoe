import json
import os
import os.path as pt

import torchvision.transforms as transforms
from torchvision.transforms import Compose

from eoe.main import ms_argsparse, multiscale_experiment, load_setup
from eoe.models.resnet import WideResNet

if __name__ == '__main__':
    def modify_parser(parser):
        parser.add_argument(
            '--magnitudes', type=int, nargs='+', default=(0, 1, 2, 4, 8, 16, 32, 64, 128, 256),
        )
        parser.add_argument(
            '--continue-run', type=str, default=None,
            help='Optional. If provided, needs to be a path to a logging directory of a previous multiscale experiment. '
                 'Providing this parameter makes the script continue the multiscale experiment by loading the setup and '
                 'results for completed magnitudes. Then, it continues the last magnitude that has not yet been completed. '
        )
        parser.set_defaults(
            comment='{obj}_imagenet_multiscale_{oesamples}OE_{msm}',
            objective='hsc',
            dataset='imagenet',
            oe_dataset='imagenet21k',
            epochs=50,
            learning_rate=1e-3,
            weight_decay=0,
            milestones=[30, 40],
            batch_size=128,
            devices=[0],
            classes=None,
            iterations=2,
        )

    def modify_args(args):
        if args.magnitude is not None:
            raise ValueError(f'The `--magnitude` argument is not used by this script as it repeats the experiment '
                             f'for all magnitudes found in `--magnitudes` instead.')
        if args.load is not None:
            raise ValueError('Since this script repeats the experiment, `--load` has no impact. Use `--continue-run` instead. ')

    args = ms_argsparse(
        lambda s: f"{s} Repeats this whole procedure multiple times with different magnitudes for the `--ms-mode`. "
                  f"This specific script comes with a default configuration for ImageNet-30. ",
        modify_parser, modify_args
    )
    args.comment = args.comment.format(
        obj=args.objective, oesamples=args.oe_size, msm="--".join(str(m).split('--')[0] for m in args.ms_mode)
    )
    train_transform = transforms.Compose([
        transforms.Resize(256),
        # transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
        transforms.RandomHorizontalFlip(p=0.5),
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
    model = WideResNet(clf=args.objective in ('bce', 'focal'))

    continue_run, last_magn_snapshots, last_magn_dir = [], None, None
    if args.continue_run is not None:
        last_magn_dir = pt.join(
            args.continue_run, sorted([d for d in os.listdir(args.continue_run) if pt.isdir(pt.join(args.continue_run, d))])[-1]
        )
        last_magn = int(pt.basename(last_magn_dir).split('_magnitude_')[-1].replace('---CNTD', ''))
        if pt.exists(pt.join(last_magn_dir, 'results.json')):
            with open(pt.join(last_magn_dir, 'results.json'), 'r') as reader:
                res = json.load(reader)
            if len([r for r in res['eval_cls_rocs'] if len(r) != 0]) == (len(args.classes) if args.classes is not None else 30):
                last_magn_dir = None
            else:
                last_magn_snapshots, _ = load_setup(last_magn_dir, args, train_transform, val_transform)
        else:
            last_magn_snapshots, _ = load_setup(last_magn_dir, args, train_transform, val_transform)
            
        previous_results = {}
        for d in sorted(os.listdir(args.continue_run)):
            if not pt.isdir(pt.join(args.continue_run, d)):
                continue
            if pt.exists(pt.join(args.continue_run, d + '---CNTD')):
                continue
            magn = int(d.split('_magnitude_')[-1].replace('---CNTD', ''))
            if 'results.json' not in os.listdir(pt.join(args.continue_run, d)):
                continue
            with open(pt.join(args.continue_run, d, 'results.json'), 'r') as reader:
                res = json.load(reader)
            if len([r for r in res['eval_cls_rocs'] if len(r) != 0]) != (len(args.classes) if args.classes is not None else 10):
                continue
            previous_results[magn] = (res['eval_mean_auc'], res['eval_std_auc'])
        assert list(previous_results.keys()) == list(args.magnitudes[:len(previous_results)]), \
            f"The so-far finished magnitudes of the loaded experiment {tuple(previous_results.keys())} " \
            f"do not match the configured magnitudes {args.magnitudes}. Please match manually."
        assert last_magn == args.magnitudes[len(previous_results)], \
            f'The last unfinished magnitude ({last_magn}) does not match the expected one in the configuration ' \
            f'[{args.magnitudes[len(previous_results)]} in {args.magnitudes}]. Please match manually. '
        continue_run = list(zip(*sorted(previous_results.items())))[1]

    print('Program started with:\n', vars(args))
    multiscale_experiment(
        args, model, train_transform, val_transform, magnitudes=args.magnitudes, continue_run=continue_run,
        continue_last_magnitude=(last_magn_snapshots, last_magn_dir), superdir=args.superdir,
    )
