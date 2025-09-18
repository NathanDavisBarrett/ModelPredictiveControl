from ...Util.Math import Array3

import numpy as np
from ...System import System


class InitializationState:
    def __init__(
        self,
        mass: float,
        position: Array3,
        velocity: Array3,
        acceleration: Array3,
        artificial_acceleration: Array3,
        thrust: Array3,
    ):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.artificial_acceleration = artificial_acceleration
        self.thrust = thrust

    @property
    def gamma(self) -> float:
        return np.linalg.norm(self.thrust)

    @property
    def artificial_acceleration_mag(self) -> float:
        return np.linalg.norm(self.artificial_acceleration)

    @classmethod
    def from_system(cls, system: System):
        return cls(
            mass=system.m,
            position=system.x,
            velocity=system.v,
            acceleration=[0.0, 0.0, 0.0],
            artificial_acceleration=[0.0, 0.0, 0.0],
            thrust=system.T,
        )
