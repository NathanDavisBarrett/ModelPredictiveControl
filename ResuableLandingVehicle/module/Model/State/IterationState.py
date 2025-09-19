"""
IterationState
==============

This class extends `InitializationState` to represent the state of the system during an iteration.

Attributes:
    dt (float): Time step duration.
    mass (float): Current mass of the system.
    prev_mass (float): Mass of the system in the previous iteration.
    gamma (float): Current magnitude of the thrust vector.
    prev_gamma (float): Magnitude of the thrust vector in the previous iteration.
    velocity (Array3): Current velocity vector of the system.
    prev_velocity (Array3): Velocity vector of the system in the previous iteration.
    wind_velocity (Array3): Wind velocity vector.
    thrust (Array3): Current thrust vector of the system.
    prev_thrust (Array3): Thrust vector of the system in the previous iteration.
    acceleration (Array3): Current acceleration vector of the system.
    prev_acceleration (Array3): Acceleration vector of the system in the previous iteration.
    position (Array3): Current position vector of the system.
    artificial_acceleration (Array3): Artificial acceleration vector of the system.

"""

from .InitializationState import InitializationState
from ...Util.Math import Array3


class IterationState(InitializationState):
    """
    IterationState
    --------------

    Extends `InitializationState` to represent the state of the system during an iteration.

    Attributes:
        dt (float): Time step duration.
        mass (float): Current mass of the system.
        prev_mass (float): Mass of the system in the previous iteration.
        gamma (float): Current magnitude of the thrust vector.
        prev_gamma (float): Magnitude of the thrust vector in the previous iteration.
        velocity (Array3): Current velocity vector of the system.
        prev_velocity (Array3): Velocity vector of the system in the previous iteration.
        wind_velocity (Array3): Wind velocity vector.
        thrust (Array3): Current thrust vector of the system.
        prev_thrust (Array3): Thrust vector of the system in the previous iteration.
        acceleration (Array3): Current acceleration vector of the system.
        prev_acceleration (Array3): Acceleration vector of the system in the previous iteration.
        position (Array3): Current position vector of the system.
        artificial_acceleration (Array3): Artificial acceleration vector of the system.
    """

    def __init__(
        self,
        dt: float,
        mass: float,
        prev_mass: float,
        gamma: float,
        prev_gamma: float,
        velocity: Array3,
        prev_velocity: Array3,
        wind_velocity: Array3,
        thrust: Array3,
        prev_thrust: Array3,
        acceleration: Array3,
        prev_acceleration: Array3,
        position: Array3,
        artificial_acceleration: Array3,
    ):
        """
        Initializes the `IterationState` with the given parameters.

        Args:
            dt (float): Time step duration.
            mass (float): Current mass of the system.
            prev_mass (float): Mass of the system in the previous iteration.
            gamma (float): Current magnitude of the thrust vector.
            prev_gamma (float): Magnitude of the thrust vector in the previous iteration.
            velocity (Array3): Current velocity vector of the system.
            prev_velocity (Array3): Velocity vector of the system in the previous iteration.
            wind_velocity (Array3): Wind velocity vector.
            thrust (Array3): Current thrust vector of the system.
            prev_thrust (Array3): Thrust vector of the system in the previous iteration.
            acceleration (Array3): Current acceleration vector of the system.
            prev_acceleration (Array3): Acceleration vector of the system in the previous iteration.
            position (Array3): Current position vector of the system.
            artificial_acceleration (Array3): Artificial acceleration vector of the system.
        """
        super().__init__(
            mass=mass,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            artificial_acceleration=artificial_acceleration,
            thrust=thrust,
        )
        self.dt = dt
        self.prev_mass = prev_mass
        self._gamma = gamma
        self.prev_gamma = prev_gamma
        self.prev_velocity = prev_velocity
        self.prev_thrust = prev_thrust
        self.prev_acceleration = prev_acceleration
        self.wind_velocity = wind_velocity

    @property
    def gamma(self) -> float:
        """
        Retrieves the current magnitude of the thrust vector.

        Returns:
            float: Current magnitude of the thrust vector.
        """
        return self._gamma
