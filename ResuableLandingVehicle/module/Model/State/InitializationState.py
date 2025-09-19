"""
InitializationState
===================

This class represents the state of the system during initialization.

Attributes:
    mass (float): Mass of the system.
    position (Array3): Position vector of the system.
    velocity (Array3): Velocity vector of the system.
    acceleration (Array3): Acceleration vector of the system.
    artificial_acceleration (Array3): Artificial acceleration vector of the system.
    thrust (Array3): Thrust vector of the system.

Properties:
    gamma (float): Magnitude of the thrust vector.
    artificial_acceleration_mag (float): Magnitude of the artificial acceleration vector.

Methods:
    from_system: Creates an `InitializationState` instance from a `System` object.
"""

from ...Util.Math import Array3

import numpy as np
from ...System import System


class InitializationState:
    """
    InitializationState
    -------------------

    Represents the state of the system during initialization.

    Attributes:
        mass (float): Mass of the system.
        position (Array3): Position vector of the system.
        velocity (Array3): Velocity vector of the system.
        acceleration (Array3): Acceleration vector of the system.
        artificial_acceleration (Array3): Artificial acceleration vector of the system.
        thrust (Array3): Thrust vector of the system.

    Properties:
        gamma (float): Magnitude of the thrust vector.
        artificial_acceleration_mag (float): Magnitude of the artificial acceleration vector.
    """

    def __init__(
        self,
        mass: float,
        position: Array3,
        velocity: Array3,
        acceleration: Array3,
        artificial_acceleration: Array3,
        thrust: Array3,
    ):
        """
        Initializes the `InitializationState` with the given parameters.

        Args:
            mass (float): Mass of the system.
            position (Array3): Position vector of the system.
            velocity (Array3): Velocity vector of the system.
            acceleration (Array3): Acceleration vector of the system.
            artificial_acceleration (Array3): Artificial acceleration vector of the system.
            thrust (Array3): Thrust vector of the system.
        """
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.artificial_acceleration = artificial_acceleration
        self.thrust = thrust

    @property
    def gamma(self) -> float:
        """
        Computes the magnitude of the thrust vector.

        Returns:
            float: Magnitude of the thrust vector.
        """
        return np.linalg.norm(self.thrust)

    @property
    def artificial_acceleration_mag(self) -> float:
        """
        Computes the magnitude of the artificial acceleration vector.

        Returns:
            float: Magnitude of the artificial acceleration vector.
        """
        return np.linalg.norm(self.artificial_acceleration)

    @classmethod
    def from_system(cls, system: System):
        """
        Creates an `InitializationState` instance from a `System` object.

        Args:
            system (System): The system object to extract state information from.

        Returns:
            InitializationState: A new instance of `InitializationState` initialized with the system's state.
        """
        return cls(
            mass=system.m,
            position=system.x,
            velocity=system.v,
            acceleration=[0.0, 0.0, 0.0],
            artificial_acceleration=[0.0, 0.0, 0.0],
            thrust=system.T,
        )
