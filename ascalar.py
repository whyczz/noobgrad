"""
Karpathy's engine.py, or pytorch's ATen
"""

import math
from functools import reduce
from collections import deque
from typing import Iterator

type ScalarLike = "Scalar" | int | float


class Scalar:
    def __init__(
        self,
        output: int | float,
        inputs: tuple["Scalar", ...] | None = None,
        op: str | None = None,
        _exp: int|float|None = None
    ):
        self.output = output

        self.grad = 0

        self._inputs = inputs 
        self._op = op

        self._exp = _exp

    def backwards(self):
        self.grad = 1
        order = self._topological_sort_inputs()
        for scalar in order:
            scalar._backwards_current_node()

    def _topological_sort_inputs(self) -> Iterator["Scalar"]:
        visited = set()
        order = []

        def dfs(node: Scalar):
            # visit first
            visited.add(node)

            # explore inputs, only unvisited
            if node._inputs:
                for neighbor in node._inputs:
                    if neighbor not in visited:
                        dfs(neighbor)

            # now all inputs are visited, and meaning the inputs are visited, and this can execute
            order.append(node)
        
        # we only have one output (the caller, no need to explore other outputs), just start with self
        dfs(self)

        # reverse order to get the nodes without dependencies (e.g. starting with the output,
        #   then the executable inputs, ...) in order
        return reversed(order)          

    def _backwards_current_node(self):
        match self._op:
            case '+':
                for parent in self._inputs:
                    parent.grad += self.grad
            case '*':
                a, b = self._inputs
                a.grad += self.grad * b.output
                b.grad += self.grad * a.output
            case '**':
                base, exp = self._inputs[0], self._exp
                base.grad += self.grad * (exp*(base.output**(exp-1)))
            case None:
                # leaf node, stop here (base case)
                pass
            case _:
                # would be nice having a ruler for slope.
                raise NotImplementedError(f"{self._op} is not implemented.")

    # Operators
    def __add__(self, other: ScalarLike) -> "Scalar":
        if not isinstance(other, Scalar):
            other = Scalar.wrap_num(other)
        return Scalar(
            output=self.output + other.output, inputs=(self, other), op="+"
        )

    def __mul__(self, other: ScalarLike) -> "Scalar":
        if not isinstance(other, Scalar):
            other = Scalar.wrap_num(other)
        return Scalar(
            output=self.output * other.output, inputs=(self, other), op="*"
        )

    def __pow__(self, other: int|float) -> "Scalar":
        return Scalar(
            output=self.output ** other, inputs=(self,), op="**", _exp=other
        )

    def __neg__(self):
        return self * -1

    def __truediv__(self, other: ScalarLike) -> "Scalar":
        return self * other**-1

    def __sub__(self, other: ScalarLike) -> "Scalar":
        return self + (-other)

    # Reflected Operators (e.g. other + self)
    def __radd__(self, other: ScalarLike) -> "Scalar":
        if not isinstance(other, Scalar):
            other = Scalar.wrap_num(other)
        return other + self

    def __rsub__(self, other: ScalarLike) -> "Scalar":
        if not isinstance(other, Scalar):
            other = Scalar.wrap_num(other)
        return other - self

    def __rmul__(self, other: ScalarLike) -> "Scalar":
        if not isinstance(other, Scalar):
            other = Scalar.wrap_num(other)
        return other * self

    def __rtruediv__(self, other: ScalarLike) -> "Scalar":
        if not isinstance(other, Scalar):
            other = Scalar.wrap_num(other)
        return other / self

    def __repr__(self):
        return f"Scalar(output={self.output})"

    @staticmethod
    def wrap_num(num: int | float) -> "Scalar":
        return Scalar(output=num)
