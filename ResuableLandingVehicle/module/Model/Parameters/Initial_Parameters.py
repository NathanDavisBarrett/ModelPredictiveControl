"""
Initial_Parameters
==================

Extends the `SystemParameters` class to include specific parameters for the initial model.

Attributes:
    w_m (float): Weight for mass in the cost function. Default is 1.0.
    w_a (float): Weight for artificial acceleration in the cost function. Default is 5e5.
"""

from ...System import SystemParameters

from dataclasses import dataclass


@dataclass
class Initial_Parameters(SystemParameters):
    """
    Initial_Parameters
    ------------------

    Extends the `SystemParameters` class to include specific parameters for the initial model.

    Attributes:
        w_m (float): Weight for mass in the cost function. Default is 1.0.
        w_a (float): Weight for artificial acceleration in the cost function. Default is 5e5.
    """

    w_m: float = 1.0  # Weight for mass in cost function
    w_a: float = 5e5  # Weight for artificial acceleration in cost function
