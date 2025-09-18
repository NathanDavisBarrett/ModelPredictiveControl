from ..Util.Math import Math, Array3
from .SystemParameters import SystemParameters

from warnings import warn
import numpy as np


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
