from typing import Iterable

from ascalar import Scalar, ScalarLike
from nn import MLP


def loss(Y: Iterable[ScalarLike], Y_pred: Iterable[Scalar]) -> Scalar:
    assert len(Y) == len(Y_pred), f"{len(Y)=} != {len(Y_pred)}, mismatched lengths for eval outputs vs prediction outputs"  
    return sum( (y_i - y_pred_i)**2 for y_i, y_pred_i in zip(Y, Y_pred)) / len(Y)

def loop(X: Iterable[ScalarLike], Y: Iterable[ScalarLike], model: MLP, learning_rate=0.1) -> MLP:
    # forwards
    Y_pred = [model(*x) for x in X]

    # reset gradients to be safe
    for p in model.parameters():
        p.grad = 0

    # backwards
    L = loss(Y, Y_pred)
    L.backwards()

    print(f"{L.output=}")

    # gradient descent
    for p in model.parameters():
        p.output -= learning_rate*p.grad
    
    return model
