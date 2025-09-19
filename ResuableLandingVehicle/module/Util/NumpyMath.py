"""
NumpyMath
=========

This module defines the `NumpyMath` class, which implements the `Math` interface using NumPy for mathematical operations.
"""

import numpy as np
from .Math import Math, Array3, Number


class NumpyMath(Math):
    """
    NumpyMath
    ---------

    Implements the `Math` interface using NumPy for mathematical operations.
    """

    @staticmethod
    def sqrt(x: Number) -> Number:
        """
        Computes the square root of a number using NumPy.

        Args:
            x (Number): The number to compute the square root of.

        Returns:
            Number: The square root of the input number.
        """
        return np.sqrt(x)

    @staticmethod
    def dot(vec1: Array3, vec2: Array3) -> Number:
        """
        Computes the dot product of two 3D vectors using NumPy.

        Args:
            vec1 (Array3): The first vector.
            vec2 (Array3): The second vector.

        Returns:
            Number: The dot product of the two vectors.
        """
        return np.dot(vec1, vec2)

    @classmethod
    def unit_vector(cls, vec: Array3) -> Array3:
        """
        Computes the unit vector of a 3D vector using NumPy.

        Args:
            vec (Array3): The vector to normalize.

        Returns:
            Array3: The unit vector.
        """
        mag = cls.norm(vec)
        if mag == 0:
            return np.array([0, 0, 0])
        return vec / mag

    @staticmethod
    def vector_add(vec1: Array3, vec2: Array3) -> Array3:
        """
        Adds two 3D vectors element-wise using NumPy.

        Args:
            vec1 (Array3): The first vector.
            vec2 (Array3): The second vector.

        Returns:
            Array3: The element-wise sum of the two vectors.
        """
        return vec1 + vec2

    @staticmethod
    def vector_scale(vec: Array3, scalar: Number) -> Array3:
        """
        Scales a 3D vector by a scalar using NumPy.

        Args:
            vec (Array3): The vector to scale.
            scalar (Number): The scalar to multiply the vector by.

        Returns:
            Array3: The scaled vector.
        """
        return scalar * vec

    @staticmethod
    def cross(vec1: Array3, vec2: Array3) -> Array3:
        """
        Computes the cross product of two 3D vectors using NumPy.

        Args:
            vec1 (Array3): The first vector.
            vec2 (Array3): The second vector.

        Returns:
            Array3: The cross product of the two vectors.
        """
        return np.cross(vec1, vec2)
