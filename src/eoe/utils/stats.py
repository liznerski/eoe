import torch
import numpy as np


class RunningStats(object):
    """ Approximates the true running mean and running std """
    def __init__(self):
        self.n = 0
        self.m = 0
        self.m2 = 0

    def add(self, x: torch.Tensor):  # x in [n, ...] where stats are calculated over n
        with torch.no_grad():
            self.n += 1
            d = (x - self.m)
            self.m += d.mean(0) / self.n
            self.m2 += ((x - self.m) * d).mean(0)

    def mean(self) -> torch.Tensor:
        return self.m if self.n >= 1 else np.nan

    def std(self) -> torch.Tensor:
        return (self.m2 / self.n).sqrt() if self.n >= 1 else np.nan
