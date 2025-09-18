from abc import ABC, abstractmethod
from typing import Annotated, Literal, Iterable
from numbers import Number

Array3 = Annotated[Iterable[Number], Literal[3]]


class Math(ABC):
    @staticmethod
    @abstractmethod
    def sqrt(x: Number) -> Number:
        pass

    @staticmethod
    @abstractmethod
    def dot(vec1: Array3, vec2: Array3) -> Number:
        pass

    @classmethod
    def norm(cls, vec: Array3) -> Number:
        return cls.sqrt(
            cls.dot(vec, vec)  # + 1e-6
        )  # Add small number to avoid ipopt problems with zero norm

    @classmethod
    @abstractmethod
    def unit_vector(vec: Array3) -> Array3:
        pass

    @staticmethod
    @abstractmethod
    def vector_add(vec1: Array3, vec2: Array3) -> Array3:
        pass

    @staticmethod
    @abstractmethod
    def vector_scale(vec: Array3, scalar: Number) -> Array3:
        pass

    @staticmethod
    @abstractmethod
    def cross(vec1: Array3, vec2: Array3) -> Array3:
        pass
