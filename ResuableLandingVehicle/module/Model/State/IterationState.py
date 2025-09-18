from .InitializationState import InitializationState
from ...Util.Math import Array3


class IterationState(InitializationState):
    def __init__(
        self,
        dt: float,
        mass: float,
        prev_mass: float,
        gamma: float,
        prev_gamma: float,
        velocity: Array3,
        prev_velocity: Array3,
        thrust: Array3,
        prev_thrust: Array3,
        acceleration: Array3,
        prev_acceleration: Array3,
        position: Array3,
        artificial_acceleration: Array3,
    ):
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

        @property
        def gamma(self) -> float:
            return self._gamma
