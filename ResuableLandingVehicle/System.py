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
    x0: Array3 = np.array([0.15, 0.25, 2.0])  # Initial position (km)
    v0: Array3 = np.array([-0.00, 0, -0.1])  # Initial velocity (km/s)
    m0: float = 15  # Mg
    T0: Array3 = np.array(
        [0, 0, 175]
    )  # Thrust vector (kN) (in line with rocket's longitudinal axis)

    xf: Array3 = np.array([0, 0, 0])  # Final position (km)
    vf: Array3 = np.array([0, 0, 0])  # Final velocity (km/s)
    nf: Array3 = np.array([0, 0, 1])  # Final surface normal vector

    # ASSUMING CONSTANT GRAVITY, AIR PRESSURE, AND AIR DENSITY
    g: Array3 = np.array([0, 0, -9.807]) / 1000  # Gravity vector (km/s^2)
    rho: float = 1.0  # Air density at sea level (kg/m^3)
    P: float = 100  # Air pressure at sea level (kPa)

    T_min: float = 100  # Minimum thrust magnitude (kN)
    T_max: float = 250  # Maximum thrust magnitude (kN)

    dTdt_max: float = 100.0  # Maximum thrust rate of change (kN/s)
    dTdt_min: float = -100.0  # Minimum thrust rate of change (kN/s)

    Cd: float = 1.0  # Drag coefficient
    Sd: float = 10.0  # Reference area for drag (m^2)

    A_nozzle: float = 0.5  # Nozzle exit area (m^2)

    max_glide_slope: float = np.radians(
        80
    )  # Maximum glide angle from horizontal (radians)
    max_tilt = np.radians(15)  # Maximum tilt angle from horizontal (radians)

    m_dry: float = 10  # Dry mass (Mg)

    I_sp: float = 300.0  # Specific impulse (s)

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
        # D (N) = -0.5 * rho * Cd * Sd * (v_mag_m_s) * v_m_s
        # D (kN) = D(N) / 1000
        # D (kN) = -0.5 * rho * Cd * Sd * (v_mag_km_s * 1000) * (v_km_s * 1000) / 1000
        # D (kN) = -0.5 * rho * Cd * Sd * v_mag_km_s * v_km_s * 1000
        factor = -0.5 * self.rho * self.Cd * self.Sd * v_mag * 1000
        return [(factor * vi) for vi in v]  # Drag force vector (kN)

    def ComputeDragForceSq(self, v_mag_sq: Number, v_sq: Array3) -> Array3:
        factor = (
            0.5 * self.rho * self.Cd * self.Sd
        ) ** 2 * v_mag_sq  # (kg / m3 * _ * m2 * km/s)^2 = 1000^2(kg / m3 * _ * m2 * m/s)^2 = 1000^2(kg / s)^2 * (N/(kg * m/s2))^2 = (1000N/(m/s))^2 = (kN/(m/s))^2

        return [(factor * vi) for vi in v_sq]  # Drag force vector (kN^2)

    def ComputeMassDepletion(self, T_Mag: Number) -> Number:
        # (Mg/s)/kN * kN - Mg/s = Mg/s
        return -self.alpha * T_Mag - self.mdot_bp  # Total mass depletion rate (Mg/s)

    @property
    def g_mag(self) -> float:
        return self.math.norm(self.g)  # km/s^2

    @property
    def alpha(self) -> float:
        # Wanted: (Mg/s)/kN

        # 1/(s * km/s^2) = s/km = s/km * ((kg * km/s^2)/kN) = (kg/s)/kN * (1000)*(kg/Mg) = 1000 (Mg/s)/kN
        #
        return 1 / (self.I_sp * self.g_mag * 1000)  # (Mg/s)/kN

    @property
    def mdot_bp(self) -> float:
        # kPa * m^2 / (s * km/s^2) = kPa * (m^2 / (1000 m/s)) = Pa * (m * s) = kg/(m * s^2) * (m * s) = kg/s = (1/1000) Mg/s
        return self.P * self.A_nozzle / (self.I_sp * self.g_mag * 1000)  # Mg/s


class System:
    def __init__(
        self,
        params: SystemParameters,
        reference_mass: float = None,
        reference_speed: float = None,
    ):
        self.params = params
        self.math = params.math

        self.reference_mass = reference_mass
        self.reference_speed = reference_speed

        self.Reset()

    def Reset(self):
        self.x = self.params.x0.copy()
        self.v = self.params.v0.copy()
        self.m = self.params.m0
        self.T = self.params.T0.copy()
        self.T_Mag = self.math.norm(self.T)

    def step(self, T: Array3, dt: float):
        T_Mag = self.math.norm(T)  # kN

        D = self.params.ComputeDragForce(self.v, v_mag=self.reference_speed)  # kN

        m = self.m if self.reference_mass is None else self.reference_mass  # Mg

        a = self.math.vector_add(
            self.math.vector_scale((self.math.vector_add(T, D)), 1 / (m * 1000)),
            self.params.g,
        )  # (kN / Mg) * (1/1000) + km/s^2 = (m/s^2 / 1000) + km/s^2 = km/s^2

        dv = self.math.vector_scale(a, dt)  # km/s^2 * s = km/s

        dx = self.math.vector_add(
            self.math.vector_scale(self.v, dt), self.math.vector_scale(a, 0.5 * dt * dt)
        )  # km/s * s + km/s^2 * s^2 = km + km = km

        dm = self.params.ComputeMassDepletion(T_Mag=T_Mag) * dt  # Mg/s * s = Mg

        self.x = self.math.vector_add(self.x, dx)  # km + km = km
        self.v = self.math.vector_add(self.v, dv)  # km/s + km/s = km/s
        self.m += dm  # Mg + Mg = Mg

        # Now check to see if any technical limits have been violated
        if self.m < self.params.m_dry:
            warn("Mass has dropped below dry mass")  #: {self.m} < {self.params.m_dry}")

        x_rel = self.math.vector_add(
            self.x, self.math.vector_scale(self.params.xf, -1)
        )  # km + km = km
        x_mag = self.math.norm(x_rel)  # km

        lhs = x_mag * np.cos(self.params.max_glide_slope)  # km * _ = km
        rhs = self.math.dot(x_rel, self.params.e_u)  # km * _ = km
        if lhs > rhs + 1e-3:  # Add small tolerance to avoid numerical issues
            warn("Glide slope limit violated")  #: {lhs} > {rhs}")

        if T_Mag > self.params.T_max:  # kN
            warn(
                "Thrust magnitude exceeds maximum limit"  #: {T_Mag} > {self.params.T_max}"
            )

        if T_Mag < self.params.T_min:  # kN
            warn(
                "Thrust magnitude below minimum limit"
            )  #: {T_Mag} < {self.params.T_min}")

        lhs = T_Mag * np.cos(self.params.max_tilt)  # kN * _ = kN
        rhs = self.math.dot(T, self.params.e_u)  # kN * _ = kN
        if lhs > rhs:
            warn("Tilt angle limit violated")  #: {lhs} > {rhs}")

        dT = self.math.vector_add(T, self.math.vector_scale(self.T, -1))  # kN + kN = kN
        dTdt = self.math.norm(dT) / dt  # kN / s = kN/s
        if dTdt > self.params.dTdt_max:  # kN/s
            warn(
                "Thrust rate of change exceeds maximum limit"  #: {dTdt} > {self.params.dTdt_max}"
            )

        if dTdt < self.params.dTdt_min:  # kN/s
            warn(
                "Thrust rate of change below minimum limit"  #: {dTdt} < {self.params.dTdt_min}"
            )

        self.T = T  # kN
        self.T_Mag = T_Mag  # kN
