import numpy as np
from dataclasses import dataclass
from ..Util.Math import Math, Array3, Number
from ..Util.PyomoMath import PyomoMath


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
