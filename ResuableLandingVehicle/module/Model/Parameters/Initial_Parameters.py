from ...System import SystemParameters

from dataclasses import dataclass


@dataclass
class Initial_Parameters(SystemParameters):
    w_m: float = 1.0  # Weight for mass in cost function
    w_a: float = 5e5  # Weight for artificial acceleration in cost function
