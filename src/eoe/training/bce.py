import torch
from tqdm import tqdm
from torch.nn.functional import binary_cross_entropy_with_logits
from torch import sigmoid
from torch.utils.data import DataLoader
from eoe.training.ad_trainer import ADTrainer


class BCETrainer(ADTrainer):
    """ implements the binary cross entropy loss to perform semi-supervised anomaly detection with outlier exposure """

    def prepare_metric(self, cstr: str, loader: DataLoader, model: torch.nn.Module, seed: int, **kwargs) -> torch.Tensor:
        return None

    def compute_anomaly_score(self, features: torch.Tensor, center: torch.Tensor, train: bool = False, **kwargs) -> torch.Tensor:
        return sigmoid(features).squeeze()

    def loss(self, features: torch.Tensor, labels: torch.Tensor, center: torch.Tensor, **kwargs) -> torch.Tensor:
        return binary_cross_entropy_with_logits(features.squeeze(), labels.float())
