import numpy as np
from numpy.typing import NDArray
from typing import Annotated, Literal, Any, Iterable
from numbers import Number
from abc import ABC, abstractmethod
from warnings import warn

Array3 = Annotated[Iterable[Number], Literal[3]]

from scipy.integrate import solve_ivp

from dataclasses import dataclass


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
        return cls.sqrt(cls.dot(vec, vec))

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


@dataclass
class SystemParameters:
    x0: Array3 = np.array([0, 0, 10_000])  # Initial position (m)
    v0: Array3 = np.array([0, 0, -500])  # Initial velocity (m/s)
    m0: float = 34_404
    T0: Array3 = np.array(
        [0, 0, 10000]
    )  # Thrust vector (N) (in line with rocket's longitudinal axis)

    xf: Array3 = np.array([0, 0, 0])  # Final position (m)
    vf: Array3 = np.array([0, 0, 0])  # Final velocity (m/s)
    nf: Array3 = np.array([0, 0, 1])  # Final surface normal vector

    # ASSUMING CONSTANT GRAVITY, AIR PRESSURE, AND AIR DENSITY
    g: Array3 = np.array([0, 0, -9.81])  # Gravity vector (m/s^2)
    rho: float = 1.225  # Air density at sea level (kg/m^3)
    P: float = 101325  # Air pressure at sea level (Pa)

    T_min: float = 1_000  # Minimum thrust magnitude (N)
    T_max: float = 1_277_000  # 934_000  # Maximum thrust magnitude (N)

    dTdt_max: float = 100_000.0  # Maximum thrust rate of change (N/s)
    dTdt_min: float = -100_000.0  # Minimum thrust rate of change (N/s)

    Cd: float = 0.5  # Drag coefficient
    Sd: float = 1.0  # Reference area for drag (m^2)

    A_nozzle: float = 0.1  # Nozzle exit area (m^2)

    max_glide_slope: float = np.pi / 6  # Maximum glide slope (radians)
    max_tilt = np.pi / 4  # Maximum tilt angle from vertical (radians)

    m_dry: float = 25_000  # Dry mass (kg)

    I_sp: float = 348.0  # Specific impulse (s)

    e_u = np.array([0, 0, 1])  # The upward-facing unit vector

    math: Math = PyomoMath  # Math library to use

    def __post_init__(self):
        assert self.m0 > self.m_dry, "Initial mass must be greater than dry mass"
        assert (
            self.T_max > self.math.norm(self.T0) > self.T_min >= 0
        ), f"Invalid thrust limits: T_min = {self.T_min}, T0 = {self.T0}, T_max = {self.T_max}"

        assert (
            self.dTdt_max > 0 > self.dTdt_min
        ), f"Invalid thrust rate limits: dTdt_min = {self.dTdt_min}, dTdt_max = {self.dTdt_max}"

    def ComputeDragForce(self, v: Array3, v_mag: Number = None) -> Array3:
        if v_mag is None:
            v_mag = self.math.norm(v)
        factor = -0.5 * self.rho * self.Cd * self.Sd * v_mag
        return [factor * vi for vi in v]  # Drag force vector (N)

    def ComputeDragForceSq(self, v_mag_sq: Number, v_sq: Array3) -> Array3:
        factor = (0.5 * self.rho * self.Cd * self.Sd) ** 2 * v_mag_sq

        return [factor * vi for vi in v_sq]  # Drag force vector (N)

    def ComputeMassDepletion(self, T_Mag: Number) -> Number:
        return -self.alpha * T_Mag - self.mdot_bp  # Total mass depletion rate (kg/s)

    @property
    def g_mag(self) -> float:
        return self.math.norm(self.g)

    @property
    def alpha(self) -> float:
        return 1 / (self.I_sp * self.g_mag)

    @property
    def mdot_bp(self) -> float:
        return self.P * self.A_nozzle / (self.I_sp * self.g_mag)


class System:
    def __init__(self, params: SystemParameters):
        self.params = params
        self.math = params.math

        self.Reset()

    def Reset(self):
        self.x = self.params.x0.copy()
        self.v = self.params.v0.copy()
        self.m = self.params.m0
        self.T = self.params.T0.copy()
        self.T_Mag = self.math.norm(self.T)

    def step(self, T: Array3, dt: float):
        T_Mag = self.math.norm(T)

        D = self.params.ComputeDragForce(self.v)

        a = self.math.vector_add(
            self.math.vector_scale((self.math.vector_add(T, D)), 1 / self.m),
            self.params.g,
        )

        dv = self.math.vector_scale(a, dt)

        dx = self.math.vector_add(
            self.math.vector_scale(self.v, dt), self.math.vector_scale(a, 0.5 * dt * dt)
        )

        dm = self.params.ComputeMassDepletion(T_Mag=T_Mag) * dt

        self.x = self.math.vector_add(self.x, dx)
        self.v = self.math.vector_add(self.v, dv)
        self.m += dm

        # Now check to see if any technical limits have been violated
        if self.m < self.params.m_dry:
            warn(f"Mass has dropped below dry mass: {self.m} < {self.params.m_dry}")

        x_rel = self.math.vector_add(self.x, self.math.vector_scale(self.params.xf, -1))
        x_mag = self.math.norm(x_rel)

        lhs = x_mag * np.cos(self.params.max_glide_slope)
        rhs = self.math.dot(x_rel, self.params.e_u)
        if lhs > rhs + 1e-4:
            warn(f"Glide slope limit violated: {lhs} > {rhs}")

        if T_Mag > self.params.T_max:
            warn(
                f"Thrust magnitude exceeds maximum limit: {T_Mag} > {self.params.T_max}"
            )

        if T_Mag < self.params.T_min:
            warn(f"Thrust magnitude below minimum limit: {T_Mag} < {self.params.T_min}")

        lhs = T_Mag * np.cos(self.params.max_tilt)
        rhs = self.math.dot(T, self.params.e_u)
        if lhs > rhs:
            warn(f"Tilt angle limit violated: {lhs} > {rhs}")

        dT = self.math.vector_add(T, self.math.vector_scale(self.T, -1))
        dTdt = self.math.norm(dT) / dt
        if dTdt > self.params.dTdt_max:
            warn(
                f"Thrust rate of change exceeds maximum limit: {dTdt} > {self.params.dTdt_max}"
            )

        if dTdt < self.params.dTdt_min:
            warn(
                f"Thrust rate of change below minimum limit: {dTdt} < {self.params.dTdt_min}"
            )

        self.T = T
        self.T_Mag = T_Mag
