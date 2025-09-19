"""
Iterate_Parameters
==================

This class extends `Initial_Parameters` to include additional parameters for iterative optimization.

Attributes:
    w_eta_dt (float): Weight for time step change in the cost function. Default is 0.1.
    w_eta_thrust (float): Weight for thrust change in the cost function. Default is 10.0.
    previous_iterate_states (Iterable[IterationState]): List of previous iteration states for each time step.
    dt_est (float): Estimated time step for the initial guess.

Methods:
    from_initial_params: Creates an instance of `Iterate_Parameters` from initial parameters and states.
"""

from .Initial_Parameters import (
    Initial_Parameters,
)
from ..State import IterationState

from dataclasses import dataclass
from typing import Iterable
import inspect


@dataclass(kw_only=True)
class Iterate_Parameters(Initial_Parameters):
    """
    Iterate_Parameters
    ------------------

    Extends the `Initial_Parameters` class to include additional parameters for iterative optimization.

    Attributes:
        w_eta_dt (float): Weight for time step change in the cost function. Default is 0.1.
        w_eta_thrust (float): Weight for thrust change in the cost function. Default is 10.0.
        previous_iterate_states (Iterable[IterationState]): List of previous iteration states for each time step.
        dt_est (float): Estimated time step for the initial guess.
    """

    w_eta_dt: float = 0.1  # Weight for time step change in cost function
    w_eta_thrust: float = 10.0  # Weight for thrust change in cost function
    previous_iterate_states: Iterable[
        IterationState
    ]  # List of previous iteration states for each time step
    dt_est: float  # Estimated time step for initial guess

    @classmethod
    def from_initial_params(
        cls,
        initParams: Initial_Parameters,
        initStates: Iterable[IterationState],
        dt_est: float,
        w_eta_dt: float = None,
        w_eta_thrust: float = None,
    ) -> "Iterate_Parameters":
        """
        Creates an instance of `Iterate_Parameters` from initial parameters and states.

        Args:
            initParams (Initial_Parameters): The initial parameters to base the iteration on.
            initStates (Iterable[IterationState]): The initial states for each time step.
            dt_est (float): Estimated time step for the initial guess.
            w_eta_dt (float, optional): Weight for time step change in the cost function. Defaults to None which will, in tern, use the default value.
            w_eta_thrust (float, optional): Weight for thrust change in the cost function. Defaults to None which will, in tern, use the default value.

        Returns:
            Iterate_Parameters: A new instance of `Iterate_Parameters` initialized with the provided values.
        """
        kwargs = {k: v for k, v in initParams.__dict__.items()}
        kwargs["dt_est"] = dt_est
        if w_eta_dt is not None:
            kwargs["w_eta_dt"] = w_eta_dt
        if w_eta_thrust is not None:
            kwargs["w_eta_thrust"] = w_eta_thrust

        kwargs["previous_iterate_states"] = initStates

        # Remove any kwargs not in the __init__ signature

        init_params = inspect.signature(cls.__init__).parameters
        valid_keys = set(init_params.keys()) - {"self"}
        kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

        return cls(**kwargs)
