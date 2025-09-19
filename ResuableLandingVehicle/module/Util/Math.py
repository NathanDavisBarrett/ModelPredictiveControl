"""
Math
====

This module defines the `Math` abstract base class, which provides an interface for mathematical operations on vectors and scalars.
"""

from abc import ABC, abstractmethod
from typing import Annotated, Literal, Iterable
from numbers import Number

Array3 = Annotated[Iterable[Number], Literal[3]]


class Math(ABC):
    """
    Math
    ----

    Abstract base class for mathematical operations on vectors and scalars.
    """

    @staticmethod
    @abstractmethod
    def sqrt(x: Number) -> Number:
        """
        Computes the square root of a number.

        Args:
            x (Number): The number to compute the square root of.

        Returns:
            Number: The square root of the input number.
        """
        pass

    @staticmethod
    @abstractmethod
    def dot(vec1: Array3, vec2: Array3) -> Number:
        """
        Computes the dot product of two 3D vectors.

        Args:
            vec1 (Array3): The first vector.
            vec2 (Array3): The second vector.

        Returns:
            Number: The dot product of the two vectors.
        """
        pass

    @classmethod
    def norm(cls, vec: Array3) -> Number:
        """
        Computes the norm (magnitude) of a 3D vector.

        Args:
            vec (Array3): The vector to compute the norm of.

        Returns:
            Number: The norm of the vector.
        """
        return cls.sqrt(
            cls.dot(vec, vec)  # + 1e-6
        )  # Add small number to avoid ipopt problems with zero norm

    @classmethod
    @abstractmethod
    def unit_vector(vec: Array3) -> Array3:
        """
        Computes the unit vector of a 3D vector.

        Args:
            vec (Array3): The vector to normalize.

        Returns:
            Array3: The unit vector.
        """
        pass

    @staticmethod
    @abstractmethod
    def vector_add(vec1: Array3, vec2: Array3) -> Array3:
        """
        Adds two 3D vectors element-wise.

        Args:
            vec1 (Array3): The first vector.
            vec2 (Array3): The second vector.

        Returns:
            Array3: The element-wise sum of the two vectors.
        """
        pass

    @staticmethod
    @abstractmethod
    def vector_scale(vec: Array3, scalar: Number) -> Array3:
        """
        Scales a 3D vector by a scalar.

        Args:
            vec (Array3): The vector to scale.
            scalar (Number): The scalar to multiply the vector by.

        Returns:
            Array3: The scaled vector.
        """
        pass

    @staticmethod
    @abstractmethod
    def cross(vec1: Array3, vec2: Array3) -> Array3:
        """
        Computes the cross product of two 3D vectors.

        Args:
            vec1 (Array3): The first vector.
            vec2 (Array3): The second vector.

        Returns:
            Array3: The cross product of the two vectors.
        """
        pass
