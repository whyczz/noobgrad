from typing import Iterable

from ascalar import Scalar, ScalarLike
from nn import MLP


def loss(y: Iterable[ScalarLike], y_pred: Iterable[Scalar]) -> Scalar:
    assert len(y) == len(y_pred), f"{len(y)=} != {len(y_pred)}, mismatched lengths for eval outputs vs prediction outputs"  
    return sum( (y_i - y_pred_i)**2 for y_i, y_pred_i in zip(y, y_pred)) / len(y)

def loop(x: Iterable[ScalarLike], y: Iterable[ScalarLike], model):
    pass
