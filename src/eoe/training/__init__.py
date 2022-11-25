from eoe.training.bce import BCETrainer
from eoe.training.hsc import HSCTrainer
from eoe.training.dsvdd import DSVDDTrainer
from eoe.training.dsad import DSADTrainer
from eoe.training.focal import FocalTrainer
from eoe.training.clip import ADClipTrainer

TRAINER = {  # maps strings to trainer classes
    'hsc': HSCTrainer, 'bce': BCETrainer, 'clip': ADClipTrainer,
    'dsvdd': DSVDDTrainer, 'dsad': DSADTrainer, 'focal': FocalTrainer
}
