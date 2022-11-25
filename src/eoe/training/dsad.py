import torch
from torch.utils.data import DataLoader
from eoe.training.ad_trainer import ADTrainer


class DSADTrainer(ADTrainer):
    """ implements the deep semi-supervised AD method to perform semi-supervised anomaly detection with outlier exposure """

    def prepare_metric(self, cstr: str, loader: DataLoader, model: torch.nn.Module, seed: int, **kwargs) -> torch.Tensor:
        return None

    def compute_anomaly_score(self, features: torch.Tensor, center: torch.Tensor, train: bool = False, **kwargs) -> torch.Tensor:
        dists = torch.sqrt(torch.norm(features, p=2, dim=1) ** 2 + 1) - 1
        scores = 1 - torch.exp(-dists)
        return scores

    def loss(self, features: torch.Tensor, labels: torch.Tensor, center: torch.Tensor, **kwargs) -> torch.Tensor:
        dists = torch.norm(features, p=2, dim=1) ** 2
        scores = dists
        losses = torch.where(labels == 0, dists, ((dists + 1e-9) ** (-1)))  # note that, in contrast to the paper, 0 marks nominal here
        return losses.mean()
