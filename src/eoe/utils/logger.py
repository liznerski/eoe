import json
import os
import os.path as pt
import re
import sys
import tarfile
import time
import warnings
from collections import deque
from copy import deepcopy
from datetime import datetime
from typing import List, Tuple, Mapping, Union, Callable
from itertools import cycle

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm


COLORS = [  # some cool distinctive colors to be used for plotting
    (224, 28, 28), (28, 224, 224), (28, 28, 224), (164, 96, 96), (96, 164, 96), (96, 96, 164),
    (128, 64, 32), (128, 32, 128), (32, 128, 128), (164, 164, 32), (255, 124, 32), (255, 124, 32),
    (124, 255, 32), (164, 64, 255), (164, 196, 124), (196, 124, 164), (124, 164, 196)
]


class ROC(object):
    def __init__(self, tpr: List[float], fpr: List[float], ths: List[float], auc: float, std: float = 0, n: int = 1):
        """
        This is a container for receiver operator characteristic curves. We use it to store the ROC plots and
        related statistics. May also be used to store the "mean" plot of multiple ROC curves.
        @param tpr: true positive rates.
        @param fpr: false positive rates.
        @param ths: thresholds.
        @param auc: the area under the curve.
        @param std: the standard deviation in case this is a "mean" plot of multiple ROC curves.
        @param n: the number of plots that was "averaged" over in case this is a "mean" plot.
        """
        self.tpr = tpr
        self.fpr = fpr
        self.ths = ths
        self.auc = auc
        self.std = std
        self.n = n

    def get_x(self):
        return self.fpr

    def get_y(self):
        return self.tpr
    
    def get_score(self):
        return self.auc


class PRC(object):
    def __init__(self, prec: List[float], rec: List[float], ths: List[float], avg_prec: float, std: float = 0, n: int = 1):
        """
        This is a container for precision-recall curves. We use it to store the PRC plots and
        related statistics. May also be used to store the "mean" plot of multiple PRC curves.
        @param prec: precision rates.
        @param rec: recall rates.
        @param ths: thresholds.
        @param avg_prec: the average precision.
        @param std: the standard deviation in case this is a "mean" plot of multiple PRC curves.
        @param n: the number of plots that was "averaged" over in case this is a "mean" plot.
        """
        self.prec = prec
        self.rec = rec
        self.ths = ths
        self.avg_prec = avg_prec
        self.std = std
        self.n = n

    def get_x(self):
        return self.rec

    def get_y(self):
        return self.prec

    def get_score(self):
        return self.avg_prec


def mean_plot(results: Union[List['ROC'], List['PRC']]) -> Union['ROC', 'PRC']:
    """
    This computes a "mean" of multiple plots (ROCs or PRCs). 
    While the mean of the score (auc or avg precision) is precise,
    the mean curve itself is rather an approximation.
    """
    if results is None or any([r is None for r in results]) or len(results) == 0:
        return None
    results = deepcopy(results)
    y, x, ths, scr = [], [], [], []
    for res in results:
        y.append(np.asarray(res.get_y()))
        x.append(np.asarray(res.get_x()))
        ths.append(np.asarray(res.ths))
        scr.append(res.get_score())

    ml = min([len(arr) for arr in ths])
    for i in range(len(ths)):
        pick = sorted(np.random.choice(len(ths[i]), size=ml, replace=False))
        y[i] = y[i][pick]
        x[i] = x[i][pick]
        ths[i] = ths[i][pick]

    y, x, ths = np.asarray(y), np.asarray(x), np.asarray(ths)
    y, x, ths = np.mean(y, axis=0), np.mean(x, axis=0), np.mean(ths, axis=0)
    n = len(scr)
    std = np.std(scr)
    scr = np.mean(scr)
    return ROC(y, x, ths, scr, std, n) if isinstance(results[0], ROC) else PRC(y, x, ths, scr, std, n)


class JsonEncoder(json.JSONEncoder):
    """ Encoder to correctly use json on numpy arrays """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, transforms.Compose):
            return obj.transforms
        else:
            return repr(obj)


class SetupEncoder(json.JSONEncoder):
    """ Encoder to correctly use json on all setup parameters """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, transforms.Compose):
            return [repr(t) for t in obj.transforms]
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            return repr(obj)


def time_format(i: float):
    """ takes a timestamp (seconds since epoch) and transforms that into a datetime string representation """
    return datetime.fromtimestamp(i).strftime('%Y%m%d%H%M%S')


class Logger(object):
    def __init__(self, logdir: str, logname: str = '', noname=False):
        """
        This logger is used to collect and store any useful information during training.
        @param logdir: The directory where all data is logged to.
        @param logname: Appends the subdirectory "log_{CURRENT_TIME_STEMP}_{logname}" to the logdir.
        @param noname: If this is true, ignore the logname parameter.
        """
        matplotlib.use('Agg')
        self.start = int(time.time())
        if noname:
            self.dir = logdir
        else:
            self.dir = pt.join(logdir, f'log_{time_format(self.start)}_{logname}')
        self.tb_writer = SummaryWriter(self.dir)

        self.__logtxtstep = 0
        self.__warnings = []
        self.__scalars = {}
        self.__active = True

    @property
    def active(self):
        return bool(self.__active)

    def print(self, msg: str, err=False):
        """ use this to print msg to the console while also logging it in a print.txt log file """
        if self.active:
            # self.tb_writer.add_text('print', msg)
            outfile = pt.join(self.dir, 'print.txt')
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            with open(outfile, 'a') as writer:
                writer.write(msg + "\n")
        print(msg, file=sys.stderr if err else sys.stdout)

    def logtxt(self, msg: str, prnt=True, tb=False):
        """ use this to log msg to a logtxt.txt logfile and, if tb is true, also to tensorboard """
        if self.active:
            if tb:
                self.tb_writer.add_text('logtxt', msg, global_step=self.__logtxtstep)
                self.__logtxtstep += 1
            outfile = pt.join(self.dir, 'logtxt.txt')
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            with open(outfile, 'a') as writer:
                writer.write(msg + "\n")
        if prnt:
            print(msg)

    def logimg(self, name: str, tensor: torch.Tensor, nrow: int = 8, pad: int = 2,
               rowheaders: List[str] = None, row_sep_at: Tuple[int, int] = (None, None), 
               colcounter: List[str] = None, maxres: int = 128, step: int = None,
               tb: bool = False, mark: List[List[int]] = None) -> np.ndarray:
        """
        Interprets a FloatTensor (n x c x h x w) as a grid of images and writes this to a png file.
        Also logs this to tensorboard if tb is true.
        @param name: the name of the png file.
        @param tensor: the tensor of images.
        @param nrow: the number of images per row in the png.
        @param pad: the amount of padding that is added inbetween images in the grid.
        @param rowheaders: a list of headers for the rows.
            Each element of the list is printed in front of its corresponding row in the png.
            The method expects less than 6 characters per header. More characters might be printed over
            the actual images. Defaults to None, where no headers are printed.
        @param row_sep_at: two integer values or empty tuple. If it contains two integers, it adds
            an additional row of zeros that act as a separator between rows. The first value describes
            the height of the separating row and the second value the position (e.g., 1 to put inbetween the
            first and second row).
        @param colcounter: a list of headers for the columns.
            Each element of the list is printed in front of its corresponding column in the png.
            Defaults to None for no column headers.
        @param maxres: downsamples tensor s.t. h=w <= maxres
        @param step: step for the tb logging.
        @param tb: whether to also log the png to tensorboard.
        @param mark: contains ids that are to be marked by a colored border. mark is a list of lists of ids.
            All images defined by the ids in a sublist are to be marked by the same (if possible unique) color.
        @return:
        """
        if tensor.size(-1) > maxres or tensor.size(-2) > maxres:
            tensor = torch.nn.functional.interpolate(tensor, size=(maxres, maxres), mode='bilinear')

        if mark is not None:
            if tensor.size(1) == 1:
                tensor = tensor.repeat(1, 3, 1, 1)
            tensor = tensor.detach().clone()
            tensor.sub_(tensor.flatten(1).min(1)[0].reshape(-1, 1, 1, 1))
            tensor.div_(tensor.flatten(1).max(1)[0].reshape(-1, 1, 1, 1))
            for m, c in zip(mark, iter(cycle(COLORS))):
                c = torch.Tensor(c).unsqueeze(1).unsqueeze(1)
                tensor[m, :, :, :1] = c
                tensor[m, :, :1, :] = c
                tensor[m, :, :, -1:] = c
                tensor[m, :, -1:, :] = c
            t = vutils.make_grid(tensor, nrow=nrow, normalize=False, padding=pad)
        else:
            t = vutils.make_grid(tensor, nrow=nrow, scale_each=True, normalize=True, padding=pad)

        t = t.transpose(0, 2).transpose(0, 1).mul(255).numpy()

        if rowheaders is not None:
            n, c, h, w = tensor.shape
            t = np.concatenate((torch.zeros(t.shape[0], int(h * 1.8), 3), t), 1)  # add black front column
            for i, head in enumerate(rowheaders):
                if len(str(head)) > 6:
                    self.warning(
                        'Header for image {} is too large, some part of it will be printed on the actual image.'.format(name)
                    )
                sc = 0.5 + 0.5 * (tensor.shape[-2] // 40)
                th = 1 + 1 * (tensor.shape[-2] // 100)
                t = cv2.putText(
                    t, str(head), (0, h - 10 * th + (h + 2) * i),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), th
                )

        if colcounter is not None:
            n, c, h, w = tensor.shape
            t = np.concatenate((torch.zeros(32, t.shape[1], 3), t), 0)  # add black front row
            for i, s in enumerate(colcounter):
                t = cv2.putText(
                    t, str(s), (w - 24 + (w + 2) * i, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1
                )

        if row_sep_at is not None and row_sep_at[0] is not None and len(row_sep_at) == 2:
            height, at = row_sep_at
            n, c, h, w = tensor.shape
            hh, ww, c = t.shape
            sep = np.zeros((height, ww, c))
            pos = (h + pad) * at + pad // 2
            t = np.concatenate([t[:pos], sep, t[pos:]]).astype(np.float32)

        img = t.astype(np.ubyte)
        if self.active:
            file = pt.join(self.dir, f'{name}_v{step}.png') if step is not None else pt.join(self.dir, f'{name}.png')
            os.makedirs(pt.dirname(file), exist_ok=True)
            cv2.imwrite(
                file, cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[-1] == 3 else img
            )
            if tb:
                t = torch.from_numpy(img).div(255).transpose(0, 1).transpose(0, 2)
                self.tb_writer.add_image(name, t, step)

        return img

    def logjson(self, name: str, dic: dict):
        """
        Writes a given dictionary to a json file in the log directory.
        Returns without impact if the size of the dictionary exceeds 10MB.
        @param name: name of the json file.
        @param dic: serializable dictionary.
        """
        if self.active:
            outfile = pt.join(self.dir, '{}.json'.format(name))
            if not pt.exists(os.path.dirname(outfile)):
                os.makedirs(os.path.dirname(outfile))
            sz = np.sum([sys.getsizeof(v) for k, v in dic.items()])
            if sz > 10000000:
                self.warning(
                    'WARNING: Could not save {}, because size of dict is {}, which exceeded 10MB!'
                    .format(pt.join(self.dir, '{}.json'.format(name)), sz),
                )
                return
            with open(outfile, 'w') as writer:
                json.dump(dic, writer, cls=JsonEncoder, indent=3)

    def snapshot(self, name: str, net: torch.nn.Module, opt: Optimizer = None, sched: _LRScheduler = None, epoch: int = None):
        """
        Writes a snapshot of the training, i.e., network weights, optimizer state, and scheduler state to a file
        in the log directory.
        @param name: name of the snapshot file.
        @param net: the neural network.
        @param opt: the optimizer used.
        @param sched: the learning rate scheduler used.
        @param epoch: the current epoch.
        """
        if self.active:
            outfile = pt.join(self.dir, 'snapshots', f'{name}.pt')
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            odic, sdic = opt.state_dict() if opt is not None else opt, sched.state_dict() if sched is not None else sched
            torch.save(
                {'net': net.state_dict(), 'opt': odic, 'sched': sdic, 'epoch': epoch}
                , outfile
            )
            return outfile
        else:
            return None

    def logsetup(self, net: torch.nn.Module, example_net_input: torch.Tensor, params: dict, step: int = 0, tb=True):
        """
        Writes a string representation of the network and all given parameters as text to a
        configuration file named config.txt in the log directory.
        Also saves a compression of the complete current code as src.tar.gz in the log directory.
        @param net: the neural network.
        @param example_net_input: some example network input to be used to create a graph viz of the model in tensorboard.
        @param params: all parameters of the training in form of a string representation (json dump of a dictionary).
        @param step: global step that is used for tb writers.
        """
        if self.active:
            if tb:
                self.tb_writer.add_text(
                    'setup',
                    "<pre>Parameter:\n\n{}\n\n\n\nNET:\n\n{}</pre>".format(json.dumps(params, indent=3, cls=SetupEncoder), net),
                    global_step=step
                )
                if example_net_input is not None:
                    self.tb_writer.add_graph(net, example_net_input)
            self.logjson('setup' if step == 0 else f'setup_v{step}', params)

            def filter(tarinfo):
                exclude = re.compile('(.*__pycache__.*)|(.*{}.*)'.format(os.sep + 'venv' + os.sep))
                if not exclude.fullmatch(tarinfo.name):
                    return tarinfo
                else:
                    return None

            if step == 0:
                outfile = pt.join(self.dir, 'src.tar.gz')
                if not pt.exists(os.path.dirname(outfile)):
                    os.makedirs(os.path.dirname(outfile))
                with tarfile.open(outfile, "w:gz") as tar:
                    root = pt.join(pt.dirname(__file__), '..')
                    tar.add(root, arcname=os.path.basename(root), filter=filter)

                self.print('Successfully saved code at {}'.format(outfile))

    def warning(self, msg: str, unique: bool = False, print: bool = True):
        """
        Writes a warning to the WARNING.log file in the log directory.
        @param s: the warning that is to be written.
        @param unique: whether a warning that has already been written is to be ignored.
        @param print: whether to additionally print the warning on the console.
        """
        if unique and msg in self.__warnings:
            return
        if print:
            self.print(msg, err=True)
        outfile = pt.join(self.dir, 'warnings.txt')
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        with open(outfile, 'a') as writer:
            writer.write(f"{msg}\n")
        self.__warnings.append(msg)

    def timeit(self, msg: str = 'Operation'):
        """
        Returns a Timer that is to be used in a `with` statement to measure the time that all operations inside
        the `with` statement took. Once the `with` statement is exited, prints the measured time together with msg.
        """
        return self.Timer(self, msg)

    class Timer(object):
        def __init__(self, logger, msg):
            self.logger = logger
            self.msg = msg
            self.start = None

        def __enter__(self):
            self.start = time.time()

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.logger.print('{} took {} seconds.'.format(self.msg, time.time() - self.start))

    def plot_many(self, results: Union[List['ROC'], List['PRC']], labels: List[str] = None, 
                  name: str = 'roc', mean: bool = True, step: int = 0) -> Union['ROC', 'PRC']:
        """
        Plots the ROCs or PRCs of different runs together in one plot and writes that to a pdf file in the log directory.
        @param results: a list of result ROCs.
        @param labels: a list of labels for the individual ROCs, defaults to [1, ...].
        @param name: the name of the pdf file.
        @param mean: whether to also plot a dotted "mean ROC/PRC".
        @param step: global_step for tensorboard.
        @return: mean plot (or None if mean is False)
        """
        assert labels == None or len(results) == len(labels), 'one label for each roc/prc'
        if results is None or all([r is None for r in results]) or len(results) == 0:
            return None
        if labels is None:
            labels = list(range(len(results)))
        results, labels = zip(*[(r, l) for r, l in zip(results, labels) if r is not None])
        legend = []
        fig = plt.figure()
        for c, res in enumerate(results):
            plt.plot(res.get_x(), res.get_y(), linewidth=0.5)
            legend.append('{} {:5.2f}% +- {:5.2f}%'.format(labels[c], res.get_score() * 100, res.std * 100))
        if mean:
            mean_res = mean_plot(results)
            plt.plot(mean_res.get_x(), mean_res.get_y(), '--', linewidth=1)
            legend.append('{} {:5.2f}% +- {:5.2f}%'.format('mean', mean_res.get_score() * 100, mean_res.std * 100))
        else:
            mean_res = None
        plt.legend(
            legend,
            fontsize=5 if len(legend) > 25 else ('xx-small' if len(legend) > 15 else 'x-small')
            )
        if self.active:
            self.tb_writer.add_figure(name, fig, global_step=step)
            outfile = pt.join(self.dir, '{}.pdf'.format(name))
            if not pt.exists(os.path.dirname(outfile)):
                os.makedirs(os.path.dirname(outfile))
            fig.savefig(outfile, format='pdf')
        plt.close()
        return mean_res

    def add_scalar(self, name: str, value: float, step: int = None, tbonly=True):
        """
        Adds a scalar value to the list of recorded values and plots them.
        @param value: scalar to be added to the plot.
        @param name: the name of the pdf file.
        @param step: step for tensorboard.
        @param tbonly: whether to log only with tb or also directly in the file system. The latter is very ineffective and might
            slow down the training as whenever a new scalar is added and new plot need to be generated and stored.
        """
        if self.active:
            self.tb_writer.add_scalar(name, value, step)
            if not tbonly:
                if name not in self.__scalars:
                    self.__scalars[name] = []
                self.__scalars[name].append(value)
                fig = plt.figure()
                plt.plot(self.__scalars[name], linewidth=0.5)
                outfile = pt.join(self.dir, '{}.pdf'.format(name))
                os.makedirs(os.path.dirname(outfile), exist_ok=True)
                fig.savefig(outfile, format='pdf')
                plt.close()

    def hist(self, normal_ascores: Tensor, anom_ascores: Tensor, name: str, step: int = 0, zoom: float = 1, density: bool = False):
        """ creates a histogram of anomaly scores for normal and anomalous samples """
        if self.active:
            fig = plt.figure()
            plt.hist(
                [normal_ascores.detach().cpu().numpy(), anom_ascores.detach().cpu().numpy()],
                np.linspace(0, zoom, 1000), cumulative=True, histtype='step', align='mid', density=density
            )
            plt.legend(['norm', 'anom'], fontsize=5)
            outfile = os.path.join(self.dir, 'hist', f'{name}_v{step}.pdf')
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            fig.savefig(outfile)
            plt.close()

    def track(self, loops: List[int], metrics: dict, descr: str):
        """ see 'class: Tracker' below """
        return Logger.Tracker(loops, metrics, descr)

    def deactivate(self):
        """ this deactivates the logger, so all function calls are ignored """
        self.__active = False

    def activate(self):
        """ this activates the logger, so that all function calls are executed again  """
        self.__active = True

    class Tracker(tqdm):
        def __init__(self, loops: List[int], metrics: Mapping[str, Union[List, Callable]], descr: str,
                     smooth: Union[bool, int]=5, **kwargs):
            """
            Simple tqdm extension that updates its description per update-call with the most recent metrics (e.g., loss) and
            is able to track nested loops, where it requires an update for each finished iteration.
            E.g.:
            >>> ep = 0
            >>> losses = []
            >>> with Tracker([2, 4], {'loss': losses, 'ep': lambda: ep}) as t:  # initialize for 2 epochs and 4 batches
            >>>     for ep in epochs:
            >>>         for x, y in loader:
            ...             ...
            >>>             losses.append(loss_of_batch)
            >>>             t.update([0, 1])
            ...         ...
            >>>         t.update([1, 0])

            @param loops: List of steps per loop.
            @param metrics: Dictionary of metrics that are updated externally. The most recent ones are printed.
                Values of the dicionary are either lists, scalars, or callables, where the latter one is merely executed,
                rather than being searched for the latest entry.
            @param descr: static description prefix.
            @param smooth: defines the window for a running average over latest entries for all metrics that are not strings.
            @param **kwargs: see original tqdm implementation.
            """
            super().__init__(total=np.sum([np.prod(loops[:i+1]) for i in range(len(loops))]), desc=descr, **kwargs)
            self.loops = loops
            self.metrics = metrics
            self.descr = descr
            self.history = {k: deque([], max(1, smooth)) for k, _ in metrics.items()}

        def update(self, steps: List[int]):
            assert len(steps) == len(self.loops), 'one update step per loop required, some can be zero'
            for k, v in self.metrics.items():
                self.history[k].append(self._get_latest(v))
            self.set_description(f'{self.descr} {"  ".join([f"{k}:"+self._smooth(v) for k, v in self.history.items()])}')
            return super().update(n=np.sum(steps)) 

        def _smooth(self, metv: List):
            if isinstance(metv[-1], str):
                return metv[-1]
            else:
                return self._stringify(np.mean([m for m in metv if m != 'None']))

        def _stringify(self, met: float) -> str:
            return f"{met:.2e}"

        def _get_latest(self, met: List): 
            while met is not None:
                if isinstance(met, str):
                    return met  # just return strings
                try:
                    met = met[-1]  # get to last element of list
                except TypeError:
                    try:
                        return self._get_latest(met())  # test if it's callable
                    except TypeError:
                        return met  # otherwise assume it's float and return 
                except IndexError:
                    try: 
                        return met.item()  # for a scalar tensor
                    except AttributeError:
                        return 'empty'  # for an empty list
            return 'None'

