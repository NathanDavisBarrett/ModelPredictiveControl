"""
PyomoMath
=========

This module defines the `PyomoMath` class, which implements the `Math` interface using Pyomo for mathematical operations.
"""

from .Math import Math, Array3, Number


class PyomoMath(Math):
    """
    PyomoMath
    ---------

    Implements the `Math` interface using Pyomo for mathematical operations.
    """

    @staticmethod
    def sqrt(x: Number) -> Number:
        """
        Computes the square root of a number using Pyomo.

        Args:
            x (Number): The number to compute the square root of.

        Returns:
            Number: The square root of the input number.
        """
        from pyomo.environ import sqrt as pysqrt

        return pysqrt(x)

    @staticmethod
    def dot(vec1: Array3, vec2: Array3) -> Number:
        """
        Computes the dot product of two 3D vectors.

        Args:
            vec1 (Array3): The first vector.
            vec2 (Array3): The second vector.

        Returns:
            Number: The dot product of the two vectors.
        """
        return sum(a * b for a, b in zip(vec1, vec2))

    @classmethod
    def unit_vector(cls, vec: Array3) -> Array3:
        """
        Computes the unit vector of a 3D vector.

        Args:
            vec (Array3): The vector to normalize.

        Returns:
            Array3: The unit vector.
        """
        mag = cls.norm(vec)
        if mag == 0:
            return [0, 0, 0]
        return [v / mag for v in vec]

    @staticmethod
    def vector_add(vec1: Array3, vec2: Array3) -> Array3:
        """
        Adds two 3D vectors element-wise.

        Args:
            vec1 (Array3): The first vector.
            vec2 (Array3): The second vector.

        Returns:
            Array3: The element-wise sum of the two vectors.
        """
        return [a + b for a, b in zip(vec1, vec2)]

    @staticmethod
    def vector_scale(vec: Array3, scalar: Number) -> Array3:
        """
        Scales a 3D vector by a scalar.

        Args:
            vec (Array3): The vector to scale.
            scalar (Number): The scalar to multiply the vector by.

        Returns:
            Array3: The scaled vector.
        """
        return [scalar * v for v in vec]

    @staticmethod
    def cross(vec1: Array3, vec2: Array3) -> Array3:
        """
        Computes the cross product of two 3D vectors.

        Args:
            vec1 (Array3): The first vector.
            vec2 (Array3): The second vector.

        Returns:
            Array3: The cross product of the two vectors.
        """
        return [
            vec1[1] * vec2[2] - vec1[2] * vec2[1],
            vec1[2] * vec2[0] - vec1[0] * vec2[2],
            vec1[0] * vec2[1] - vec1[1] * vec2[0],
        ]
