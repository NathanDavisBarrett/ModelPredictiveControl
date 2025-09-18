import numpy as np
from .Math import Math, Array3, Number

class NumpyMath(Math):
    @staticmethod
    def sqrt(x: Number) -> Number:
        return np.sqrt(x)

    @staticmethod
    def dot(vec1: Array3, vec2: Array3) -> Number:
        return np.dot(vec1, vec2)

    @classmethod
    def unit_vector(cls, vec: Array3) -> Array3:
        mag = cls.norm(vec)
        if mag == 0:
            return np.array([0, 0, 0])
        return vec / mag

    @staticmethod
    def vector_add(vec1: Array3, vec2: Array3) -> Array3:
        return vec1 + vec2

    @staticmethod
    def vector_scale(vec: Array3, scalar: Number) -> Array3:
        return scalar * vec

    @staticmethod
    def cross(vec1: Array3, vec2: Array3) -> Array3:
        return np.cross(vec1, vec2)