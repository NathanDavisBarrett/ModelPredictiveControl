from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np


class RandomParameter(ABC):
    @abstractmethod
    def sample(self):
        pass


class UniformRandomParameter(RandomParameter):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self):
        return np.random.uniform(self.low, self.high)


class NormalRandomParameter(RandomParameter):
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

    def sample(self):
        return np.random.normal(self.mean, self.stddev)


@dataclass
class SystemParameters:
    T_max = 400  # C

    T_jacket_bounds = (250, 350)  # K
    T_jacket_0 = 305  # K
    T_jacket_ramp_rate = 5  # K / sec

    Vdot = NormalRandomParameter(100, 1)  # m^3 / sec
    V = 100  # m^3

    rho = NormalRandomParameter(1000, 4)  # kg/m^3
    Cp = NormalRandomParameter(0.239, 0.01)  # 0.239  # J / kg K

    dH_rxn = 5e4  # J / mol

    R = 8.314  # J / mol K
    E = 8750 * R  # J / mol

    k0 = 7.2e10  # 1 / sec (pre-exponential factor)
    UA = NormalRandomParameter(
        5e4, 1000
    )  # 5e4  # J / sec m^2 K (overall heat transfer coefficient * area)

    C_in = NormalRandomParameter(1, 0.05)  # mol / m^3 (inlet concentration)
    T_in = UniformRandomParameter(345, 355)  # C (inlet temperature)

    C_0 = 0.9  # mol / m^3 (initial control-volume concentration)
    T_0 = 305  # K  (initial control-volume temperature)


class System:
    def __init__(self, params: SystemParameters = None):
        if params is None:
            params = SystemParameters()
        self.params = params
        self.C = params.C_0
        self.T = params.T_0
        self.T_jacket = params.T_jacket_0

    def step(self, T_jacket, dt):
        T_jacket *= np.random.uniform(0.99, 1.01)
        # Sample random parameters
        Vdot = self.params.Vdot.sample()
        rho = self.params.rho.sample()
        Cp = self.params.Cp.sample()
        UA = self.params.UA.sample()
        C_in = self.params.C_in.sample()
        T_in = self.params.T_in.sample()

        k = self.params.k0 * np.exp(-self.params.E / (self.params.R * self.T))
        rate = k * self.C  # Reaction rate

        dCdt = (Vdot / self.params.V) * (C_in - self.C) - rate
        dTdt = (
            (Vdot / self.params.V) * (T_in - self.T)
            + (self.params.dH_rxn / (rho * Cp)) * rate
            + (UA / (self.params.V * rho * Cp)) * (T_jacket - self.T)
        )

        # Euler integration
        self.C += dCdt * dt
        self.T += dTdt * dt
        self.T_jacket = T_jacket

        return self.C, self.T, self.T_jacket
