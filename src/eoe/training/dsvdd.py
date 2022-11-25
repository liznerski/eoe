import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from eoe.training.ad_trainer import ADTrainer


class DSVDDTrainer(ADTrainer):
    """ implements deep support vector description to perform unsupervised anomaly detection """

    def prepare_metric(self, cstr: str, loader: DataLoader, model: torch.nn.Module, seed: int, **kwargs) -> torch.Tensor:
        center = []
        eps = kwargs.get('eps', 1e-1)
        for imgs, lbls, _ in tqdm(loader, desc=f'cls {cstr} preparing DSVDD center'):
            imgs = imgs.to(self.device)
            with torch.no_grad():
                image_features = model(imgs[lbls == 0])
            center.append(image_features.cpu().mean(0).unsqueeze(0))
        center = torch.cat(center).mean(0).unsqueeze(0).to(self.device)
        center[(abs(center) < eps) & (center < 0)] = -eps
        center[(abs(center) < eps) & (center > 0)] = eps
        return center

    def compute_anomaly_score(self, features: torch.Tensor, center: torch.Tensor, train: bool = False, **kwargs) -> torch.Tensor:
        return (features - center).pow(2).sum(-1)

    def loss(self, features: torch.Tensor, labels: torch.Tensor,  center: torch.Tensor, **kwargs) -> torch.Tensor:
        return (features - center).pow(2).sum(-1).mean()
