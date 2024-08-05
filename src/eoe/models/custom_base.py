from abc import ABC, abstractmethod

import torch.nn as nn


class CustomNet(nn.Module, ABC):
    @abstractmethod
    def __init__(self, feature_model_output_dim: int, prediction_head: bool = True, clf: bool = False,
                 freeze: bool = False):
        """
        @param feature_model_output_dim: the dimensionality of the flattened output of the feature model.
        @param prediction_head: whether to add an additional final linear prediction head with
            either 256 output neurons (HSC, ...) or 1 output neuron (BCE, focal, ...).
        @param clf: whether this model is to be used for a classification objective (BCE, focal, ...), which requires
            one output neuron.
        @param freeze: whether to make the trainer call :meth:`freeze_parts` before training.
        """
        super().__init__()
        self.feature_model: nn.Module = nn.Identity()  # implement this
        self.feature_dim = feature_model_output_dim
        self.clf = clf
        self.prediction_head = prediction_head
        self.freeze = freeze

        if self.prediction_head:
            self.final_linear = nn.Linear(self.feature_dim, 1 if self.clf else 256)
        elif self.clf and feature_model_output_dim != 1:
            raise ValueError(
                f"{self.__class__} was created for a classification loss (BCE, focal, ...). "
                f"However, it was created without an additional prediction head and its feature model predicts more "
                f"than one neuron ({self.feature_dim} > 1). The loss won't work! Please use an "
                f"additional prediction head (--custom-model-add-prediction-head) or change the objective."
            )

    def freeze_parts(self, ) -> bool:
        if self.freeze:
            for n, p in self.feature_model.named_parameters():
                p.requires_grad_(False)
            return True
        return False

    def load_feature_model_weights(self, model_state_dict: dict):
        self.feature_model.load_state_dict(model_state_dict)

    def forward(self, x):
        features = self.feature_model(x)
        if self.prediction_head:
            out = self.final_linear(features.flatten(1))
        else:
            out = features
        return out
