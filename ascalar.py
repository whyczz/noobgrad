"""
Karpathy's engine.py, or pytorch's ATen
"""
import math

type ScalarLike = "Scalar" | int | float

class Scalar:
    def __init__(self, value: int | float):
        self.value = value

    def __eq__(self, other):
        return math.isclose(self.value, other.value)

    # Operators
    def __add__(self, other: ScalarLike) -> "Scalar":
        if not isinstance(other, Scalar):
            other = Scalar.wrap_num(other)
        return Scalar(value=self.value + other.value)

    def __mul__(self, other: ScalarLike) -> "Scalar":
        if not isinstance(other, Scalar):
            other = Scalar.wrap_num(other)
        return Scalar(value=self.value * other.value)
    
    def __pow__(self, other: ScalarLike) -> "Scalar":
        if not isinstance(other, Scalar):
            other = Scalar.wrap_num(other)
        return Scalar(value=self.value ** other.value)

    def __neg__(self):
        return Scalar(value=-self.value)

    def __truediv__(self, other: ScalarLike) -> "Scalar":
        return Scalar(value=self.value * other.value**-1)
    
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

    def __rpow__(self, other: ScalarLike) -> "Scalar":
        if not isinstance(other, Scalar):
            other = Scalar.wrap_num(other)
        return other ** self

    def __repr__(self):
        return f"Scalar(value={self.value})"

    @staticmethod
    def wrap_num(num: int | float) -> "Scalar":
        return Scalar(value=num)