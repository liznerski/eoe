import torch
from tqdm import tqdm
from torch.nn.functional import binary_cross_entropy_with_logits
from torch import sigmoid
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from eoe.training.ad_trainer import ADTrainer


# Implementation of the focal loss from https://arxiv.org/abs/1708.02002
class FocalLoss(_Loss):
    """ implements the focal loss to perform semi-supervised anomaly detection with outlier exposure """

    def __init__(self, gamma=2.0, eps=1e-7):
        super(FocalLoss, self).__init__(size_average=None, reduce=None, reduction='mean')
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        BCE_loss = binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        pt = pt.clamp(self.eps, 1. - self.eps)
        F_loss = (1 - pt).pow(self.gamma) * BCE_loss
        return F_loss.mean()


class FocalTrainer(ADTrainer):
    def prepare_metric(self, cstr: str, loader: DataLoader, model: torch.nn.Module, seed: int, **kwargs) -> torch.Tensor:
        return None

    def compute_anomaly_score(self, features: torch.Tensor, center: torch.Tensor, train: bool = False, **kwargs) -> torch.Tensor:
        return sigmoid(features).squeeze()

    def loss(self, features: torch.Tensor, labels: torch.Tensor, center: torch.Tensor, **kwargs) -> torch.Tensor:
        return FocalLoss()(features.squeeze(), labels.float())
