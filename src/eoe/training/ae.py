import torch
from torch.utils.data import DataLoader
from eoe.training.ad_trainer import ADTrainer


class AETrainer(ADTrainer):
    """ implements a reconstructive loss to perform unsupervised anomaly detection """

    def prepare_metric(self, cstr: str, loader: DataLoader, model: torch.nn.Module, seed: int, **kwargs) -> torch.Tensor:
        return None

    def compute_anomaly_score(self, features: torch.Tensor, center: torch.Tensor, train: bool = False, **kwargs) -> torch.Tensor:
        return (features - kwargs['inputs']).pow(2).flatten(1).sum(-1)

    def loss(self, features: torch.Tensor, labels: torch.Tensor,  center: torch.Tensor, **kwargs) -> torch.Tensor:
        return (features - kwargs['inputs']).pow(2).flatten(1).sum(-1).mean()
