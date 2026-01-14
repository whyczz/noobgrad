"""
from torch.nn import ?
from noobgrad.nn import ?
"""

import random
import math
from typing import Iterable 
from functools import reduce

from ascalar import Scalar, ScalarLike

def unwrap(x: Iterable[ScalarLike]) -> ScalarLike:
    if len(x) == 1:
        return x[0]
    return x

class Neuron:
    def __init__(self, n_in: int):  # n_in = num_inputs
        self.w = [Scalar(random.uniform(-1, 1)) for _ in range(n_in)] # weights
        self.b = Scalar(random.uniform(-1, 1)) # bias
    
    def forward(self, *x: ScalarLike) -> ScalarLike:  # x = inputs
        assert len(x) == len(self.w), f"mismatched num_inputs {len(self.w)=} and input length {len(x)=}"

        dot_prod = 0
        for w_, x_ in zip(self.w, x):
            dot_prod += w_*x_
        
        linear = dot_prod + self.b
        output = linear.tanh()
        return output

    def parameters(self) -> Iterable[Scalar]:
        return [*self.w, self.b] 
    
    def __call__(self, *x: ScalarLike) -> ScalarLike:
        return self.forward(*x)

    def __str__(self) -> str:
        return f"Neuron({self.w=}, {self.b=})"


class Layer:
    def __init__(self, n_in: int, n_out: int):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]
    
    def forward(self, *x: ScalarLike) -> Iterable[Scalar]:
        outputs = [n(*x) for n in self.neurons]
        return outputs

    def parameters(self) -> Iterable[Scalar]:
        return [param for n in self.neurons for param in n.parameters()] 

    def __call__(self, *x: ScalarLike) -> Iterable[Scalar]:
        return self.forward(*x)

    def __str__(self) -> str:
        s = "Layer(\n"
        for neuron in self.neurons:
            s += (str(neuron) + "\n")
        s += ")"
        return s


class MLP:
    def __init__(self, sizes: Iterable[int]):
        self.layers = [
            Layer(n_in, n_out)
            for n_in, n_out in zip(sizes, sizes[1:])
        ]
    
    def forward(self, *x: ScalarLike) -> Iterable[ScalarLike] | ScalarLike:
        for layer in self.layers:
            x = layer(*x)
        return unwrap(x)

    def parameters(self) -> Iterable[Scalar]:
        return [param for layer in self.layers for param in layer.parameters()]

    def __call__(self, *x: ScalarLike) -> Iterable[ScalarLike] | ScalarLike:
        return self.forward(*x)
    
    def __str__(self) -> str:
        s = "MLP(\n"
        for layer in self.layers:
            s += (str(layer) + "\n")
        s += ")"
        return s




