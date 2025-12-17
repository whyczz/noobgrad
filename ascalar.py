"""
Karpathy's engine.py, or pytorch's ATen
"""
import math

class Scalar:
    def __init__(self, value: int | float):
        self.value = value

    def __eq__(self, other):
        return math.isclose(self.value, other.value)

    # Operators
    def __add__(self, other: "Scalar") -> "Scalar":
        return Scalar(value=self.value + other.value)

    def __mul__(self, other: "Scalar") -> "Scalar":
        return Scalar(value=self.value * other.value)
    
    def __pow__(self, other: int | float) -> "Scalar":
        return Scalar(value=self.value ** other)

    def __neg__(self):
        return Scalar(value=-self.value)

    def __truediv__(self, other: "Scalar") -> "Scalar":
        return Scalar(value=self.value * other.value**-1)
    
    def __sub__(self, other: "Scalar") -> "Scalar":
        return self + (-other)

    # Reflected Operators (e.g. other + self)

    def __repr__(self):
        return f"Scalar(value={self.value})"