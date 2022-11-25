from abc import abstractmethod, ABC
from copy import deepcopy
from typing import Union, List, Tuple, Generic, TypeVar

import numpy as np
import torch
from sklearn.metrics import auc as compute_auc, roc_curve, precision_recall_curve, average_precision_score
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

from eoe.datasets import load_dataset, str_labels, no_classes, MSM
from eoe.datasets.bases import TorchvisionDataset, CombinedDataset
from eoe.datasets.imagenet import ADImageNet21k
from eoe.models.clip_official.clip.model import CLIP
from eoe.utils.logger import Logger, ROC, PRC


class NanGradientsError(RuntimeError):
    pass


def lst_of_lsts(n: int):  # initiates a list of n empty lists
    return [[] for _ in range(n)]


def weight_reset(m: torch.nn.Module):  # resets the weights of the given module
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


R = TypeVar('R')


class Result(Generic[R]):
    def __init__(self, classes: int):
        """
        Used to track metrics for all classes and random seeds.
        Result consists of a list (len = #classes) with each element again being a list (len = #seeds) of recorded metrics.
        Via __getitem__() one can access the recorded metrics for a specific class.
        Additionally, one can add a mean result for a class via set_mean(), which can be accessed later via
        mean() and means().
        E.g.:
        >>> rocs = Result(len(classes))
        >>> for cls in classes:
        >>>     for seed in seeds:
        >>>         ...
        >>>         rocs[cls].append(training_result_roc)
        >>>     ...
        >>>     rocs.set_mean(cls, mean_plot(rocs[cls]))
        >>> plot_many(rocs.means())
        """
        self.values = lst_of_lsts(classes)
        self.mean_values = [None] * classes

    def __getitem__(self, cls: int) -> List[R]:
        """ return the recorded metrics for the class cls """
        return self.values[cls]

    def set_mean(self, cls: int, value: R):
        """ set the mean for the class cls (e.g., a mean ROC plot) """
        self.mean_values[cls] = value

    def mean(self, cls: int, on_none_return_latest=False) -> R:
        """
        @param cls: determines the class whose mean is to be returned.
        @param on_none_return_latest: whether to return the recorded metric for the latest seed or None for missing means.
        @return: the set mean.
        """
        mean = self.mean_values[cls]
        latest = self.values[cls][-1] if len(self.values[cls]) > 0 else None
        return mean if mean is not None else (latest if on_none_return_latest else None)

    def means(self, on_none_return_latest=False) -> List[R]:
        """ returns a list of all set means """
        return [self.mean(cls, on_none_return_latest) for cls in range(len(self.mean_values))]

    def __str__(self) -> str:
        return str(self.values)

    def __repr__(self) -> str:
        return repr(self.values)

    def __iter__(self):
        return iter(self.values)


class ADTrainer(ABC):
    AD_MODES = ('one_vs_rest', 'leave_one_out')  # all possible AD benchmark modes
    # whether to keep the model snapshots in RAM and make the `run` method return them in addition to storing them on the disk
    KEEP_SNAPSHOT_IN_RAM = False

    def __init__(self, model: torch.nn.Module, train_transform: Compose, test_transform: Compose,
                 dataset: str, oe_dataset: str, datapath: str, logger: Logger,
                 epochs: int, lr: float, wdk: float, milestones: List[int], batch_size: int,
                 ad_mode: str = 'one_vs_rest', device: Union[str, torch.device] = 'cuda',
                 oe_limit_samples: int = np.infty, oe_limit_classes: int = np.infty,
                 msms: List[MSM] = (), workers: int = 2):
        """
        The base trainer class.
        It defines a `run` method that iterates over all classes and multiple random seeds per class.
        For each class-seed combination, it trains and evaluates a given AD model.
        The objective needs to be implemented (see :method:`loss`, etc.); this is an abstract class.
        Pre-implemented trainers with objectives can be found in other files; e.g., :class:`eoe.training.hsc.HSCTrainer`.
        Depending on the ad_mode, the trainer either treats the current class or all but the current class as normal.
        The trainer always evaluates using the full test set.
        For training, it uses only normal samples and perhaps auxiliary anomalies from a different source (Outlier Exposure).
        For a list of all available training configurations, have a look at the parameters below.

        @param model: some model that is to be trained/evaluated. For training multiple classes/seeds, a separate
            copy of the model is initialized and trained.
        @param train_transform: pre-processing pipeline applied to training samples (included data augmentation).
        @param test_transform: pre-processing pipeline applied to test samples (including data augmentation).
        @param dataset: string specifying the dataset, see :data:`eoe.datasets.__init__.DS_CHOICES`.
        @param oe_dataset: string specifying the Outlier Exposure dataset, see :data:`eoe.datasets.__init__.DS_CHOICES`.
        @param datapath: filepath to where the datasets are located or automatically to be downloaded to.
            Specifies the root directory for all datasets.
        @param logger: a logger instance that is used to print current training progress and log metrics to the disk.
        @param epochs: how many full iterations of the dataset are to be trained per class-seed combination.
        @param lr: initial learning rate.
        @param wdk: weight decay.
        @param milestones: milestones for the learning rate scheduler; at each milestone the learning rate is reduced by 0.1.
        @param batch_size: batch size. If there is an OE dataset, the overall batch size will be twice as large
            as an equally-sized batch of OE samples gets concatenated to the batch of normal training samples.
        @param ad_mode: anomaly detection mode; either 'one_vs_all', where the current class is considered normal
            and the rest anomalous, or 'leave_one_out', where the current class is considered anomalous and the rest normal.
        @param device: torch device to be used. 'cuda' uses the first available GPU.
        @param oe_limit_samples: limits the number of different samples for the OE dataset (randomly selects a subset of samples).
        @param oe_limit_classes: limits the number of different classes for the OE dataset (randomly selects a subset of classes).
        @param msms: a list of MSMs (multi-scale modes). Each MSM contains a transformation and a type of data.
            Each MSM is passed to the datasets so that they apply the transformation to the corresponding data type.
            For instance, the MSM could be "apply LowPassFilter to NormalTrainingData".
            See :class:`eoe.datasets.__init__.MSM`.
            For usual training, MSMs should not be used as they utilize labels to transform samples.
            We used it to experiment with different version of frequency filters in our frequency analysis experiments, however,
            have only reported results for equal filters on all data types in the paper.
        @param workers: number of data-loading workers. See :class:`torch.utils.data.DataLoader`.
        """
        logger.logsetup(model, None, {k: v for k, v in locals().items() if k not in ['self', 'model']})  # log training setup
        self.model = model.cpu() if model is not None else model
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.dsstr = dataset
        self.oe_dsstr = oe_dataset
        self.oe_limit_samples = oe_limit_samples
        self.oe_limit_classes = oe_limit_classes
        self.msms = msms
        self.datapath = datapath
        self.logger = logger
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.wdk = wdk
        self.milestones = milestones
        self.batch_size = batch_size
        self.ad_mode = ad_mode
        self.center = None
        self.workers = workers
        self.ds = None

    def get_nominal_classes(self, cur_class: int):
        # get the set of normal classes for the current class depending on the AD mode
        if self.ad_mode == 'one_vs_rest':
            return [cur_class]
        elif self.ad_mode == 'leave_one_out':
            return [c for c in range(no_classes(self.dsstr)) if c != cur_class]
        elif self.ad_mode == 'fifty_fifty':
            return [c % no_classes(self.dsstr) for c in range(cur_class, no_classes(self.dsstr) // 2 + cur_class)]
        else:
            raise NotImplementedError(f'AD mode {self.ad_mode} unknown. Known modes are {ADTrainer.AD_MODES}.')

    def run(self, run_classes: List[int] = None, run_seeds: int = 1,
            load: List[List[Union[Module, str]]] = None) -> Tuple[List[List[Module]], dict]:
        """
        Iterates over all classes and multiple random seeds per class.
        For each class-seed combination, it trains and evaluates a given AD model using the trainer's loss
        (see method:`ADTrainer.loss`). For example, see :class:`eoe.training.hsc` for an implementation of the HSC loss.
        This method also prepares the complete dataset (all splits and perhaps outlier exposure)
        at the start of the training of each class-seed combination.

        @param run_classes: which classes to run on. If none, iterates over all classes of the given dataset.
        @param run_seeds: how often to train a class with different random seeds.
        @param load: a list (len = #classes) with each element again being a list (len = #seeds) of model snapshots.
            The snapshots can either be PyTorch Module instances defining the model directly or strings specifying
            filepaths to stored snapshots in the form of dictionaries. These contain the model, optimizer, and scheduler state.
            Whenever the trainer starts training with the j-th random seed for class i, it tries to initialize the model
            with the snapshot found in load[i][j]. If not available, it trains a model from scratch.
            The model snapshots need to match the model architecture specified in the initialization of the trainer.
            If the snapshots are dictionaries, the trainer just continues training at the stored last epoch and, if the number
            of overall epochs has been reached already, just evaluates the model.
        @return: returns a tuple of
            - a list (len = #classes) with each element again being a list (len = #seeds) of trained AD models.
              If ADTrainer.KEEP_SNAPSHOT_IN_RAM is False, this list will be None to prevent out-of-memory errors.
            - a dictionary containing all important final metrics recorded at the end of the training and during
              evaluation.
            All the returned information is always also stored on the disk using the trainer's logger.
        """
        self.logger.logsetup(None, None, {'run_classes': run_classes, 'run_seeds': run_seeds, 'load': load}, step=1)

        """ trains and evaluates cls-wise """
        # prepare variables
        classes = str_labels(self.dsstr)
        run_classes = run_classes if run_classes is not None else list(range(len(classes)))
        train_cls_rocs = Result(len(classes))
        eval_cls_rocs = Result(len(classes))
        eval_cls_prcs = Result(len(classes))
        models = lst_of_lsts(len(classes))
        assert self.ds is None or len(run_classes) == 1, \
            'pre-loading DS (setting trainer.ds to something) only allowed for one class'

        # Loop over all classes, considering in each step the current class nominal
        for c, cstr in ((c, cstr) for c, cstr in enumerate(classes) if c in run_classes):

            for seed in range(run_seeds):
                self.logger.print(f'------ start training cls {c} "{cstr}" ------')
                if load is not None and len(load) > c and len(load[c]) > seed:
                    cur_load = load[c][seed]
                else:
                    cur_load = None

                # prepare model
                def copy_model():
                    if cur_load is not None and isinstance(cur_load, Module):
                        self.logger.print('Loaded model (not snapshot).')
                        model = deepcopy(cur_load)
                    else:
                        model = deepcopy(self.model)
                        if not isinstance(model, CLIP):
                            model.apply(weight_reset)
                    assert all([p.is_leaf for p in self.model.parameters()])
                    for n, p in model.named_parameters():
                        p.detach_().requires_grad_()  # otherwise jit models don't work due to grad_fn=clone_backward
                    return model

                orig_cache_size = ADImageNet21k.img_cache_size
                if isinstance(cur_load, str) and self.load_epochs_only(cur_load) >= self.epochs:
                    ADImageNet21k.img_cache_size = 0
                ds = load_dataset(
                    self.dsstr, self.datapath, self.get_nominal_classes(c), 0,
                    self.train_transform, self.test_transform, self.logger, self.oe_dsstr,
                    self.oe_limit_samples, self.oe_limit_classes, self.msms
                ) if self.ds is None else self.ds
                ADImageNet21k.img_cache_size = orig_cache_size

                # train
                for i in range(5):
                    try:
                        model = copy_model()
                        model, roc = self.train_cls(model, ds, c, cstr, seed, cur_load)
                        break
                    except NanGradientsError as err:  # try once more
                        self.logger.warning(
                            f'Gradients got NaN for class {c} "{cstr}" and seed {seed}. '
                            f'Happened {i} times so far. Try once more.'
                        )
                        ds = load_dataset(
                            self.dsstr, self.datapath, self.get_nominal_classes(c), 0,
                            self.train_transform, self.test_transform, self.logger, self.oe_dsstr,
                            self.oe_limit_samples, self.oe_limit_classes, self.msms
                        ) if self.ds is None else self.ds
                        if i == 3 - 1:
                            model, roc = None, None
                            self.logger.warning(
                                f'Gradients got NaN for class {c} "{cstr}" and seed {seed}. '
                                f'Happened {i} times so far. Try no more. Set model and roc to None.'
                            )
                models[c].append(model)
                train_cls_rocs[c].append(roc)
                self.logger.plot_many(
                    train_cls_rocs.means(True), classes, name='training_intermediate_roc', step=c*run_seeds+seed
                )

                # eval 
                model = models[c][-1]
                if model is not None:
                    roc, prc = self.eval_cls(model, ds, c, cstr, seed)
                else:
                    roc, prc = None, None
                eval_cls_rocs[c].append(roc)
                eval_cls_prcs[c].append(prc)
                self.logger.plot_many(eval_cls_rocs.means(True), classes, name='eval_intermediate_roc', step=c*run_seeds+seed)
                self.logger.plot_many(eval_cls_prcs.means(True), classes, name='eval_intermediate_prc', step=c*run_seeds+seed)

                if model is not None:
                    self.logger.snapshot(f'snapshot_cls{c}_it{seed}', model, epoch=self.epochs)
                    if not ADTrainer.KEEP_SNAPSHOT_IN_RAM:
                        models[c][-1] = None

                del ds

            # seed-wise many_roc plots for current class 
            cls_mean_roc = self.logger.plot_many(train_cls_rocs[c], None, name=f'training_cls{c}-{cstr}_roc', step=c)
            train_cls_rocs.set_mean(c, cls_mean_roc)        
            cls_mean_roc = self.logger.plot_many(eval_cls_rocs[c], None, name=f'eval_cls{c}-{cstr}_roc', step=c)
            eval_cls_rocs.set_mean(c, cls_mean_roc)
            cls_mean_prc = self.logger.plot_many(eval_cls_prcs[c], None, name=f'eval_cls{c}-{cstr}_prc', step=c)
            eval_cls_prcs.set_mean(c, cls_mean_prc)

        # training: compute cls-wise roc curves and combine in a final overview roc plot
        if any([t is not None for t in train_cls_rocs.means()]):
            mean_auc = np.mean([m.auc for m in train_cls_rocs.means() if m is not None]) 
            std_auc = np.std([m.auc for m in train_cls_rocs.means() if m is not None])
            self.logger.logtxt(f'Training: Overall {mean_auc*100:04.2f}% +- {std_auc*100:04.2f} AUC.')
            self.logger.plot_many(train_cls_rocs.means(), classes, name='training_roc')

            # print an overview of cls-wise rocs
            print('--------------- OVERVIEW ------------------')
            for auc, cstr in ((a.auc, c) for a, c in zip(train_cls_rocs.means(), classes) if a is not None):
                print(f'Training: Class "{cstr}" yields {auc*100:04.2f}% AUC.')
            print(f'Training: Overall {mean_auc*100:04.2f}% +- {std_auc*100:04.2f} AUC.')

        # evaluation: compute cls-wise roc curves and combine in a final overview roc plot
        mean_auc = np.mean([m.auc for m in eval_cls_rocs.means() if m is not None]) 
        std_auc = np.std([m.auc for m in eval_cls_rocs.means() if m is not None])
        self.logger.plot_many(eval_cls_rocs.means(), classes, name='eval_roc')
        mean_avg_prec = np.mean([m.avg_prec for m in eval_cls_prcs.means() if m is not None]) 
        std_avg_prec = np.std([m.avg_prec for m in eval_cls_prcs.means() if m is not None])
        self.logger.plot_many(eval_cls_prcs.means(), classes, name='eval_prc')

        # print some overview of the achieved scores
        self.logger.logtxt('--------------- OVERVIEW ------------------')
        self.logger.logtxt(f'Eval: Overall {mean_avg_prec*100:04.2f}% +- {std_avg_prec*100:04.2f}% AvgPrec.')
        for auc, std, cstr in ((a.auc, a.std, c) for a, c in zip(eval_cls_rocs.means(), classes) if a is not None):
            self.logger.logtxt(f'Eval: Class "{cstr}" yields {auc*100:04.2f}% +- {std*100:04.2f}% AUC.')
        self.logger.logtxt(f'Eval: Overall {mean_auc*100:04.2f}% +- {std_auc*100:04.2f}% AUC.')

        self.logger.logjson('results', {
            'eval_mean_auc': mean_auc, 'eval_std_auc': std_auc, 'eval_mean_avg_prec': mean_avg_prec,
            'eval_cls_rocs': [[roc.get_score() if roc is not None else None for roc in cls_roc] for cls_roc in eval_cls_rocs],
            'classes': classes
        })
        return models, {
            'mean_auc': mean_auc, 'mean_avg_prec': mean_avg_prec, 'std_auc': std_auc,
            'cls_aucs': [[roc.get_score() if roc is not None else None for roc in cls_roc] for cls_roc in eval_cls_rocs]
        }

    def train_cls(self, model: torch.nn.Module, ds: TorchvisionDataset, cls: int, clsstr: str, seed: int,
                  load: Union[Module, str] = None) -> Tuple[torch.nn.Module, ROC]:
        """
        Trains the given model for the current class.
        @param model: the AD model that is to be trained.
        @param ds: the dataset containing normal training samples and perhaps Outlier Exposure.
            If it contains OE, the dataset is an instance of a CombinedDataset (see :class:`eoe.datasets.bases.CombinedDataset`).
            The loader of a combined dataset returns a batch where the first half are normal training samples and
            the second half is made of Outlier Exposure.
        @param cls: the current class. E.g., for the one vs. rest benchmark, this is the normal class.
        @param clsstr: A string representation for the current class (e.g., 'airplane' for ds being CIFAR-10 and class being 0).
        @param seed: the current iteration of random seeds. E.g., `2` denotes the second random seed of the current class.
        @param load: if not None, initializes the AD model with `load`. `load` can either be a PyTorch module or a filepath.
            If it is a filepath, also loads the last epoch with which the stored model was trained and only trains
            for the remaining number of epochs. The architecture found in `load` needs to match the one specified in the
            trainer's initialization.
        @return: the trained model and training ROC.
        """
        # ---- prepare model and variables
        model = model.to(self.device).train()
        epochs = self.epochs
        cls_roc = None

        # ---- optimizers and loaders
        if isinstance(model, CLIP):
            opt = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.wdk, momentum=0.9, nesterov=True)
        else:
            opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wdk, amsgrad=False)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, self.milestones, 0.1)
        loader, _ = ds.loaders(self.batch_size, num_workers=self.workers, persistent=True)
        if seed == 0 and self.logger.active:
            prev = ds.preview(40, True,  classes=[0, 1] if isinstance(ds, CombinedDataset) else [0])
            stats = ds.n_normal_anomalous()
            self.logger.logimg(
                f'training_cls{cls}-{clsstr}_preview', prev, nrow=prev.shape[0] // len(stats),
                rowheaders=[str(stats[k]) for k in sorted(stats.keys())]
            )
            del prev

        # ---- prepare trackers and loggers  
        ep, loss = self.load(load if isinstance(load, str) else None, model, opt, sched), None
        center = self.center = self.prepare_metric(clsstr, loader, model, seed)
        to_track = {
            'ep': lambda: f'{ep+1:{len(str(epochs))}d}/{epochs}', 'loss': lambda: loss.item() if loss is not None else None, 
            'roc': lambda: cls_roc.auc if cls_roc is not None else None, 'lr': lambda: sched.get_last_lr()[0]
        }

        with self.logger.track([epochs, len(loader)], to_track, f'training cls{cls}') as tracker:

            # ---- loop over epochs
            for ep in range(ep, epochs):
                ep_labels, ep_ascores = [], [] 
                
                # ---- loop over batches
                for imgs, lbls, idcs in loader:
                    imgs = imgs.to(self.device)
                    lbls = lbls.to(self.device)
                    with torch.no_grad():
                        if isinstance(ds, CombinedDataset):
                            imgs[lbls == ds.nominal_label] = ds.normal.gpu_train_conditional_transform(
                                imgs[lbls == ds.nominal_label], [ds.nominal_label] * len(imgs[lbls == ds.nominal_label])
                            )
                            imgs[lbls == ds.nominal_label] = ds.normal.gpu_train_transform(imgs[lbls == ds.nominal_label])
                            imgs[lbls != ds.nominal_label] = ds.oe.gpu_train_conditional_transform(
                                imgs[lbls != ds.nominal_label], [ds.anomalous_label] * len(imgs[lbls != ds.nominal_label])
                            )
                            imgs[lbls != ds.nominal_label] = ds.oe.gpu_train_transform(imgs[lbls != ds.nominal_label])
                        else:
                            imgs = ds.gpu_train_conditional_transform(imgs, lbls)
                            imgs = ds.gpu_train_transform(imgs)

                    # ---- compute loss and optimize
                    opt.zero_grad()
                    image_features = model(imgs)
                    loss = self.loss(image_features, lbls, center, inputs=imgs)
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
                    anomaly_scores = self.compute_anomaly_score(image_features, center, inputs=imgs).cpu()

                    # ---- log stuff
                    ep_labels.append(lbls.detach().cpu())
                    ep_ascores.append(anomaly_scores.detach().cpu())
                    self.logger.add_scalar(
                        f'training_cls{cls}_seed{seed}_loss', loss.item(), tracker.n,
                    )
                    tracker.update([0, 1])

                # ---- prepare labels and anomaly scores of epoch
                ep_labels, ep_ascores = torch.cat(ep_labels), torch.cat(ep_ascores)
                if ep_ascores.isnan().sum() > 0:
                    raise NanGradientsError()

                # ---- compute training AuROC
                if (ep_labels == 1).sum() > 0:
                    fpr, tpr, thresholds = roc_curve(ep_labels, ep_ascores.squeeze())
                    auc = compute_auc(fpr, tpr)
                    cls_roc = ROC(tpr, fpr, thresholds, auc)

                # ---- log epoch stuff
                self.logger.tb_writer.add_histogram(
                    f'Training: CLS{cls} SEED{seed} anomaly_scores normal', ep_ascores[ep_labels == 0], ep,
                )
                if (ep_labels == 1).sum() > 0:
                    self.logger.tb_writer.add_histogram(
                        f'Training: CLS{cls} SEED{seed} anomaly_scores anomalous', ep_ascores[ep_labels == 1], ep,
                    )
                    self.logger.add_scalar(f'Training: CLS{cls} SEED{seed} AUC', cls_roc.auc*100, ep, )
                
                # ---- update tracker and scheduler
                sched.step()
                tracker.update([1, 0])

        return model.cpu().eval(), cls_roc

    def eval_cls(self, model: torch.nn.Module, ds: TorchvisionDataset, cls: int, clsstr: str, seed: int) -> Tuple[ROC, PRC]:
        """
        Evaluates the given model for the current class.
        Returns and logs the ROC and PRC metrics.
        @param model: the (trained) model to be evaluated.
        @param ds: the dataset to be used for evaluating (should be a test split of some dataset).
        @param cls: the current class. E.g., for the one vs. rest benchmark, this is the normal class.
        @param clsstr: A string representation for the current class (e.g., 'airplane' for ds being CIFAR-10 and class being 0).
        @param seed: the current iteration of random seeds. E.g., `2` denotes the second random seed of the current class.
        @return: ROC and PRC metric.
        """
        model = model.to(self.device).eval()
        _, loader = ds.loaders(self.batch_size, num_workers=self.workers, shuffle_test=False)
        if seed == 0 and self.logger.active:
            prev = ds.preview(20, False)
            stats = ds.n_normal_anomalous(False)
            self.logger.logimg(
                f'eval_cls{cls}-{clsstr}_preview', prev, nrow=prev.shape[0] // 2,
                rowheaders=[str(stats[0]), str(stats[1])]
            )
            del prev

        center = self.center
        ep_labels, ep_ascores = [], []  # [...], list of all labels/etc.
        procbar = tqdm(desc=f'evaluating cls {clsstr}', total=len(loader))
        for imgs, lbls, idcs in loader:
            imgs = imgs.to(self.device)
            if isinstance(ds, CombinedDataset):
                imgs = ds.normal.gpu_test_conditional_transform(imgs, lbls)
                imgs = ds.normal.gpu_test_transform(imgs)
            else:
                imgs = ds.gpu_test_conditional_transform(imgs, lbls)
                imgs = ds.gpu_test_transform(imgs)
            with torch.no_grad():
                image_features = model(imgs)
            anomaly_scores = self.compute_anomaly_score(image_features, center, inputs=imgs)
            ep_labels.append(lbls.cpu())
            ep_ascores.append(anomaly_scores.cpu())
            procbar.update()
        procbar.close()
        ep_labels, ep_ascores = torch.cat(ep_labels), torch.cat(ep_ascores)

        fpr, tpr, thresholds = roc_curve(ep_labels, ep_ascores.squeeze())
        auc = compute_auc(fpr, tpr)
        cls_roc = ROC(tpr, fpr, thresholds, auc)

        prec, rec, thresholds = precision_recall_curve(ep_labels, ep_ascores.squeeze())
        average_prec = average_precision_score(ep_labels, ep_ascores.squeeze())
        cls_prc = PRC(prec, rec, thresholds, average_prec)

        self.logger.logtxt(
            f'Eval: class "{clsstr}" yields {auc * 100:04.2f}% AUC and {average_prec * 100:04.2f}% average precision (seed {seed}).'
        )
        self.logger.tb_writer.add_histogram(
            f'Eval: (SD{seed}) anomaly_scores cls{cls} nominal', ep_ascores[ep_labels == 0], 0, walltime=0
        )
        self.logger.tb_writer.add_histogram(
            f'Eval: (SD{seed}) anomaly_scores cls{cls} anomalous', ep_ascores[ep_labels == 1], 0, walltime=0
        )
        model.cpu()

        return cls_roc, cls_prc

    def load(self, path: str, model: torch.nn.Module,
             opt: torch.optim.Optimizer = None, sched: _LRScheduler = None) -> int:
        """
        Loads a snapshot of the model including training state.
        @param path: the filepath where the snapshot is stored.
        @param model: the model instance into which the parameters of the found snapshot are loaded.
            Hence, the architectures need to match.
        @param opt: the optimizer instance into which the training state is loaded.
        @param sched: the learning rate scheduler into which the training state is loaded.
        @return: the last epoch with which the snapshot's model was trained.
        """
        epoch = 0
        if path is not None:
            snapshot = torch.load(path)
            net_state = snapshot.pop('net', None)
            opt_state = snapshot.pop('opt', None)
            sched_state = snapshot.pop('sched', None)
            epoch = snapshot.pop('epoch', 0)
            if net_state is not None:
                model.load_state_dict(net_state)
            if opt_state is not None and opt is not None:
                opt.load_state_dict(opt_state)
            if sched_state is not None and sched is not None:
                sched.load_state_dict(sched_state)
            self.logger.print(f'Loaded snapshot at epoch {epoch}')
        return epoch

    def load_epochs_only(self, path: str):
        """ loads the last epoch with which the snapshot's model found at `path` was trained """
        if path is None:
            return 0
        else:
            return torch.load(path).pop('epoch', 0)

    @abstractmethod
    def prepare_metric(self, cstr: str, loader: DataLoader, model: torch.nn.Module, seed: int, **kwargs) -> torch.Tensor:
        """
        Implement a 'center' (DSVDD) or, in general, a reference tensor for the anomaly score metric.
        Executed at the beginning of training (even if training epochs == 0).
        Optional for Outlier Exposure-based methods.
        @param cstr: the string representation of the current class (e.g., 'airplane' for ds being CIFAR-10 and class being 0).
            For the one vs. rest benchmark, the current class is the normal class.
        @param loader: a data loader that can be used to compute the reference tensor.
            The trainer's `train_cls` method executes `prepare_metric` and passes the training loader for this purpose.
        @param model: The model for which the reference tensor is to be computed.
        @param seed: the current iteration of random seeds. E.g., `2` denotes the second random seed of the current class.
        @param kwargs: potential further implementation-specific parameters.
        @return: the reference tensor.
        """
        pass

    @abstractmethod
    def compute_anomaly_score(self, features: torch.Tensor, center: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Implement a method that computes the anomaly scores for a given batch of image features.
        @param features: a batch of image features (shape: n x d). The trainer computes these features with the AD model.
        @param center: a center or, in general, a reference tensor (shape: d) that can be used to compute the anomaly scores.
        @param kwargs: potential further implementation-specific parameters.
        @return: the batch of anomaly scores (shape: n).
        """
        pass

    @abstractmethod
    def loss(self, features: torch.Tensor, labels: torch.Tensor,  center: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Implement a method that computes the loss for a given batch of image features.
        @param features: a batch of image features (shape: n x d). The trainer computes these features with the AD model.
        @param labels: a batch of corresponding integer labels (shape: n).
        @param center: a center or, in general, a reference tensor (shape: d) that can be used to compute the anomaly scores.
        @param kwargs: potential further implementation-specific parameters.
        @return: the loss (scalar).
        """
        pass
