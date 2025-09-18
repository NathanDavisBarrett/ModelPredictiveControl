from .Math import Math, Array3, Number

class PyomoMath(Math):
    @staticmethod
    def sqrt(x: Number) -> Number:
        from pyomo.environ import sqrt as pysqrt

        return pysqrt(x)

    @staticmethod
    def dot(vec1: Array3, vec2: Array3) -> Number:
        return sum(a * b for a, b in zip(vec1, vec2))

    @classmethod
    def unit_vector(cls, vec: Array3) -> Array3:
        mag = cls.norm(vec)
        if mag == 0:
            return [0, 0, 0]
        return [v / mag for v in vec]

    @staticmethod
    def vector_add(vec1: Array3, vec2: Array3) -> Array3:
        return [a + b for a, b in zip(vec1, vec2)]

    @staticmethod
    def vector_scale(vec: Array3, scalar: Number) -> Array3:
        return [scalar * v for v in vec]

    @staticmethod
    def cross(vec1: Array3, vec2: Array3) -> Array3:
        return [
            vec1[1] * vec2[2] - vec1[2] * vec2[1],
            vec1[2] * vec2[0] - vec1[0] * vec2[2],
            vec1[0] * vec2[1] - vec1[1] * vec2[0],
        ]