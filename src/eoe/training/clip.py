import torch
import eoe.models.clip_official.clip as official_clip
from torchvision import transforms
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from eoe.training.ad_trainer import ADTrainer
from eoe.datasets import str_labels


def raise_error(t: str):
    raise ValueError(f'transform placeholder {t} is unknown.')


class ADClipTrainer(ADTrainer):

    def __init__(self, model: torch.nn.Module, train_transform: Compose, test_transform: Compose, *args,
                 anom_tkn_ptn='a photo of something', **kwargs, ):
        """
        Implements CLIP for unsupervised and semi-supervised AD with outlier exposure.
        For the pre-trained CLIP architecture, we use an older release of the official repository in `eoe.models.clip_official`.
        @param model: has to be None. Always uses the CLIP model.
        @param train_transform: has to be None. Always uses CLIP's transform.
        @param test_transform: has to be None. Always uses CLIP's transform.
        @param args: further args (see ADTrainer).
        @param anom_tkn_ptn: the text associated with the anomalous class. Normal classes will always be associated
            with "a photo of a NORMAL_CLASS_NAME". `anom_tkn_ptn` can contain a {} which is replaced with the NORMAL_CLASS_NAME.
        @param kwargs: further kwargs (see ADTrainer).
        """
        assert model is None, 'CLIP-AD always uses the CLIP model'
        assert test_transform is None or len(test_transform.transforms) == 0, "CLIP-AD always uses CLIP's test transform"
        print('Load CLIP model.')
        model, transform = official_clip.load('ViT-B/32', 'cuda', jit=False)
        model.forward = model.encode_image
        if train_transform is not None:
            train_transform.transforms = [
                t if not isinstance(t, str) else (
                    transforms.Compose(transform.transforms[:3]) if t == 'clip_pil_preprocessing' else (
                        transform.transforms[-1] if t == 'clip_tensor_preprocessing' else
                        raise_error(t)
                    )
                )
                for t in train_transform.transforms
            ]
        else:
            train_transform = transform
        super(ADClipTrainer, self).__init__(None, train_transform, transform, *args, **kwargs)
        self.model = model.to(self.device)
        self.anom_tkn_ptn = anom_tkn_ptn

    def prepare_metric(self, cstr: str, loader: DataLoader, model: torch.nn.Module, seed: int, **kwargs) -> torch.Tensor:
        if self.ad_mode == 'one_vs_rest':
            raw_texts = [f"a photo of a {cstr}", self.anom_tkn_ptn.format(cstr)]
        elif self.ad_mode == 'leave_one_out':
            raw_texts = [*[f"a photo of a {cs}" for cs in str_labels(self.dsstr) if cs != cstr], self.anom_tkn_ptn.format(cstr)]
        else:
            raise NotImplementedError()
        self.raw_texts = raw_texts

        text_inputs = torch.cat([official_clip.tokenize(tk) for tk in raw_texts]).to(self.device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def compute_anomaly_score(self, image_features: torch.Tensor, center: torch.Tensor,
                              train: bool = False, **kwargs) -> torch.Tensor:
        with torch.no_grad() if not train else torch.enable_grad():
            text_features = center / center.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            # certainties, predictions = similarity.topk(1)
            if self.ad_mode == 'one_vs_rest':
                anomaly_scores = similarity[:, -1]
            elif self.ad_mode == 'leave_one_out':
                anomaly_scores = similarity[:, -1]
            else:
                raise NotImplementedError()
        return anomaly_scores

    def loss(self, image_features: torch.Tensor, labels: torch.Tensor, center: torch.Tensor, **kwargs) -> torch.Tensor:
        text_features = center
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).log_softmax(dim=-1)
        if self.ad_mode == 'one_vs_rest':
            aloss = similarity[labels == 1][:, -1]
            nloss = similarity[labels == 0][:, 0]
            loss = torch.zeros_like(similarity[:, 0])
            loss[labels == 1] = aloss
            loss[labels == 0] = nloss
        elif self.ad_mode == 'leave_one_out':
            aloss = similarity[labels == 1][:, -1]
            nloss = similarity[labels == 0][:, :-1].max(-1)[0]
            loss = torch.zeros_like(similarity[:, 0])
            loss[labels == 1] = aloss
            loss[labels == 0] = nloss
        else:
            raise NotImplementedError()
        loss = loss.mul(-1)
        loss = loss.mean()
        return loss
