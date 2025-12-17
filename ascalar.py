"""
Karpathy's engine.py, or pytorch's ATen
"""

class Scalar:
    def __init__(self, value: int | float):
        self.value = value

    def __repr__(self):
        return f"Scalar(value={self.value})"