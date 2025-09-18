from typing import Union
import pyomo.kernel as pmo
from ...Util.Math import Array3


class State:
    def __init__(
        self,
        mass: Union[float, pmo.variable],
        position: Array3,
        velocity: Array3,
        acceleration: Array3,
        thrust: Array3,
        gamma: Union[float, pmo.variable],
    ):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.thrust = thrust
        self.gamma = gamma
