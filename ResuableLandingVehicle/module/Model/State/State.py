"""
State
=====

This class represents the state of the system at a given time step.

Attributes:
    mass (Union[float, pmo.variable]): Mass of the system, which can be a fixed value or a Pyomo variable.
    position (Array3): Position vector of the system.
    velocity (Array3): Velocity vector of the system.
    acceleration (Array3): Acceleration vector of the system.
    thrust (Array3): Thrust vector of the system.
    gamma (Union[float, pmo.variable]): Magnitude of the thrust vector, which can be a fixed value or a Pyomo variable.
"""

from typing import Union
import pyomo.kernel as pmo
from ...Util.Math import Array3


class State:
    """
    State
    -----

    Represents the state of the system at a given time step.

    Attributes:
        mass (Union[float, pmo.variable]): Mass of the system, which can be a fixed value or a Pyomo variable.
        position (Array3): Position vector of the system.
        velocity (Array3): Velocity vector of the system.
        acceleration (Array3): Acceleration vector of the system.
        thrust (Array3): Thrust vector of the system.
        gamma (Union[float, pmo.variable]): Magnitude of the thrust vector, which can be a fixed value or a Pyomo variable.
    """

    def __init__(
        self,
        mass: Union[float, pmo.variable],
        position: Array3,
        velocity: Array3,
        acceleration: Array3,
        thrust: Array3,
        gamma: Union[float, pmo.variable],
    ):
        """
        Initializes the `State` with the given parameters.

        Args:
            mass (Union[float, pmo.variable]): Mass of the system, which can be a fixed value or a Pyomo variable.
            position (Array3): Position vector of the system.
            velocity (Array3): Velocity vector of the system.
            acceleration (Array3): Acceleration vector of the system.
            thrust (Array3): Thrust vector of the system.
            gamma (Union[float, pmo.variable]): Magnitude of the thrust vector, which can be a fixed value or a Pyomo variable.
        """
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.thrust = thrust
        self.gamma = gamma
