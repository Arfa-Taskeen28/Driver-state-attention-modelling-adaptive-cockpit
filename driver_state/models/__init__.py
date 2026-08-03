from .lstm import LSTMClassifier
from .transformer import TransformerClassifier

MODELS = {"lstm": LSTMClassifier, "transformer": TransformerClassifier}

__all__ = ["LSTMClassifier", "TransformerClassifier", "MODELS"]
