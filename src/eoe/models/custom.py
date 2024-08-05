from eoe.models.custom_base import CustomNet
from eoe.models.resnet import WideResNet


class WideResNetCustom(CustomNet):
    def __init__(self, prediction_head: bool = True, clf: bool = False, freeze: bool = False):
        super().__init__(256, prediction_head, clf, freeze)
        self.feature_model = WideResNet(256, False)

    # ### Per default, if self.freeze is True, CustomNet freezes gradients for the entire feature model.
    # ### For a different behavior, reimplement:
    # def freeze_parts(self,) -> bool:

    # ### Per default, CustomNet extracts features using self.feature_model, passes those through a prediction
    # ### head if self.prediction_head is True, and returns the outcome.
    # ### For a different behavior, reimplement:
    # def forward(self, x):


# Here, implement your own custom models, inheriting from CustomNet.
# They will automatically become available for :file:`eoe.main.train_only_custom` and :file:`eoe.main.inference_custom`.


